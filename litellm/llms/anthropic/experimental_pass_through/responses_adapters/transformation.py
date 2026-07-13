"""
Transformation layer: Anthropic /v1/messages <-> OpenAI Responses API.

This module owns all format conversions for the direct v1/messages -> Responses API
path used for OpenAI and Azure models.
"""

import base64
import binascii
import hashlib
import json
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.litellm_core_utils.reasoning_effort_utils import (
    reasoning_effort_from_thinking_budget,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    truncate_tool_name,
)
from litellm.llms.anthropic.experimental_pass_through.utils import (
    is_reasoning_auto_summary_enabled,
)
from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthopicMessagesAssistantMessageParam,
    AnthropicFinishReason,
    AnthropicMessagesRequest,
    AnthropicMessagesToolChoice,
    AnthropicMessagesUserMessageParam,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockThinking,
    AnthropicResponseContentBlockToolUse,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from litellm.types.llms.openai import ResponsesAPIResponse

OPENAI_MIN_RESPONSE_OUTPUT_TOKENS = 16


def _token_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    return value if isinstance(value, int) and value > 0 else 0


def _is_billing_header_block(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    block = cast(Dict[str, object], value)  # cast-ok: guarded by the dict check above
    text = block.get("text")
    return block.get("type") == "text" and isinstance(text, str) and text.startswith("x-anthropic-billing-header:")


def _filter_billing_headers_from_system(system: object) -> Optional[Union[str, List[object]]]:
    if isinstance(system, str):
        return None if system.startswith("x-anthropic-billing-header:") else system
    if not isinstance(system, list):
        return None
    blocks = cast(List[object], system)  # cast-ok: guarded by the list check above
    return [block for block in blocks if not _is_billing_header_block(block)]


class _ReasoningSignaturePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    encrypted_content: str


class _ReasoningSummaryData(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    text: str


class _ReasoningOutputItem(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    type: Literal["reasoning"]
    id: str
    encrypted_content: Optional[str] = None
    summary: tuple[_ReasoningSummaryData, ...] = ()


class _ContextManagementEdit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    keep: object = None


class _ContextManagementInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    edits: tuple[_ContextManagementEdit, ...] = ()


_REASONING_SIGNATURE_PREFIX = "litellm_openai_reasoning_v1_"
_MAX_REASONING_PAYLOAD_BYTES = 8 * 1024 * 1024
_MAX_REASONING_ITEM_ID_BYTES = 1024
_MAX_ENCRYPTED_CONTENT_BYTES = _MAX_REASONING_PAYLOAD_BYTES - 4096
_MAX_REASONING_SIGNATURE_CHARS = len(_REASONING_SIGNATURE_PREFIX) + 4 * ((_MAX_REASONING_PAYLOAD_BYTES + 2) // 3)


def encode_reasoning_signature(
    item_id: object,
    encrypted_content: object,
) -> Optional[str]:
    if not isinstance(item_id, str) or not item_id:
        return None
    if not isinstance(encrypted_content, str) or not encrypted_content:
        return None
    if len(item_id.encode()) > _MAX_REASONING_ITEM_ID_BYTES:
        return None
    if len(encrypted_content.encode()) > _MAX_ENCRYPTED_CONTENT_BYTES:
        return None
    payload = json.dumps(
        {"encrypted_content": encrypted_content, "id": item_id},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if len(payload) > _MAX_REASONING_PAYLOAD_BYTES:
        return None
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"{_REASONING_SIGNATURE_PREFIX}{encoded}"


def encode_reasoning_item_signature(item: object) -> Optional[str]:
    reasoning_item = _parse_reasoning_output_item(item)
    if reasoning_item is None:
        return None
    return encode_reasoning_signature(reasoning_item.id, reasoning_item.encrypted_content)


def _parse_reasoning_output_item(item: object) -> Optional[_ReasoningOutputItem]:
    try:
        return _ReasoningOutputItem.model_validate(item)
    except ValidationError:
        return None


def decode_reasoning_signature(signature: object) -> Optional[_ReasoningSignaturePayload]:
    if not isinstance(signature, str) or not signature.startswith(_REASONING_SIGNATURE_PREFIX):
        return None
    if len(signature) > _MAX_REASONING_SIGNATURE_CHARS:
        return None
    encoded = signature[len(_REASONING_SIGNATURE_PREFIX) :]
    if not encoded or re.fullmatch(r"[A-Za-z0-9_-]+", encoded) is None:
        return None
    try:
        raw_payload = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
        if len(raw_payload) > _MAX_REASONING_PAYLOAD_BYTES:
            return None
        payload = _ReasoningSignaturePayload.model_validate_json(raw_payload)
    except (binascii.Error, ValidationError, ValueError):
        return None
    if encode_reasoning_signature(payload.id, payload.encrypted_content) != signature:
        return None
    return payload


def _context_management_clears_thinking(context_management: object) -> bool:
    try:
        parsed = _ContextManagementInput.model_validate(context_management)
    except ValidationError:
        return False
    return any(edit.type == "clear_thinking_20251015" and edit.keep == "none" for edit in parsed.edits)


# OpenAI Responses service_tier values -> Anthropic service_tier values.
_SERVICE_TIER_MAP = {
    "default": "standard",
    "auto": "standard",
    "flex": "standard",
    "scale": "standard",
    "priority": "priority",
}


def map_service_tier(value: object) -> Optional[str]:
    """Map an OpenAI Responses service_tier to the closest Anthropic value.

    Unknown values pass through unchanged so we never invent data; ``None`` is
    returned when the response carried no service tier.
    """
    if not isinstance(value, str) or not value:
        return None
    return _SERVICE_TIER_MAP.get(value, value)


def normalize_stop_sequences(stop_sequences: object) -> Tuple[str, ...]:
    """Return the non-empty string stop sequences from an arbitrary value."""
    if not isinstance(stop_sequences, (list, tuple)):
        return ()
    return tuple(s for s in stop_sequences if isinstance(s, str) and s)


def find_earliest_stop_sequence(text: str, stop_sequences: Sequence[str]) -> Optional[Tuple[int, str]]:
    """Return ``(index, sequence)`` of the earliest stop-sequence match in ``text``.

    The Responses API has no native stop parameter, so stop sequences are emulated
    by scanning generated text. Ties on index are broken by the longest sequence so
    truncation is deterministic.
    """
    best: Optional[Tuple[int, str]] = None
    for seq in stop_sequences:
        if not seq:
            continue
        idx = text.find(seq)
        if idx == -1:
            continue
        if best is None or idx < best[0] or (idx == best[0] and len(seq) > len(best[1])):
            best = (idx, seq)
    return best


def partial_stop_suffix_len(buffer: str, stop_sequences: Sequence[str]) -> int:
    """Length of the longest suffix of ``buffer`` that is a proper prefix of a stop sequence.

    Those trailing characters must be withheld while streaming because a later delta
    could grow them into a full stop-sequence match.
    """
    max_len = 0
    for seq in stop_sequences:
        if not seq:
            continue
        limit = min(len(buffer), len(seq) - 1)
        for k in range(limit, max_len, -1):
            if buffer[-k:] == seq[:k]:
                max_len = k
                break
    return max_len


def apply_stop_sequences_to_content(
    content: List[Dict[str, Any]],
    stop_sequences: Sequence[str],
) -> Optional[str]:
    """Truncate ``content`` in place at the first stop-sequence hit in a text block.

    Mirrors Anthropic semantics: the stop sequence and everything after it (including
    any later content blocks) is removed and the matched sequence is returned. Returns
    ``None`` when no stop sequence appears in the output text.
    """
    for i, block in enumerate(content):
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text", "")
        if not isinstance(text, str):
            continue
        match = find_earliest_stop_sequence(text, stop_sequences)
        if match is None:
            continue
        idx, seq = match
        block["text"] = text[:idx]
        del content[i + 1 :]
        return seq
    return None


_TOOL_USE_ERROR_OPEN = "<tool_use_error>"
_TOOL_USE_ERROR_CLOSE = "</tool_use_error>"


def _wrap_tool_error(output: object) -> object:
    """Wrap an errored tool_result payload in Claude Code's <tool_use_error> marker."""
    if isinstance(output, str):
        return f"{_TOOL_USE_ERROR_OPEN}{output}{_TOOL_USE_ERROR_CLOSE}"
    if isinstance(output, list):
        parts = cast(List[Dict[str, object]], output)  # cast-ok: built as input_text/input_image parts above
        wrapped_any = False
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "input_text":
                text = part.get("text")
                part["text"] = f"{_TOOL_USE_ERROR_OPEN}{text if isinstance(text, str) else ''}{_TOOL_USE_ERROR_CLOSE}"
                wrapped_any = True
        if not wrapped_any:
            parts.insert(0, {"type": "input_text", "text": f"{_TOOL_USE_ERROR_OPEN}{_TOOL_USE_ERROR_CLOSE}"})
        return parts
    return output


# JSON Schema keywords the OpenAI Responses API rejects on function parameters.
_UNSUPPORTED_SCHEMA_ROOT_KEYS = ("$schema", "$id", "$anchor", "$comment", "id")


def sanitize_tool_parameters(input_schema: object) -> Dict[str, Any]:
    """Coerce an Anthropic input_schema into a Responses-compatible parameters object.

    Guarantees an object-typed schema and strips root keywords the Responses API
    rejects, so a nonconforming tool degrades instead of 400-ing the whole request.
    """
    if not isinstance(input_schema, dict):
        return {"type": "object", "properties": {}}
    schema = {k: v for k, v in cast(Dict[str, Any], input_schema).items() if k not in _UNSUPPORTED_SCHEMA_ROOT_KEYS}
    if schema.get("type") != "object":
        schema["type"] = "object"
    if not isinstance(schema.get("properties"), dict):
        schema["properties"] = {}
    return schema


# Anthropic hosted tools with no Responses equivalent (web_search / code_execution
# are handled explicitly and are not listed here).
_UNSUPPORTED_HOSTED_TOOL_NAMES = frozenset(
    {"bash", "text_editor", "computer", "memory", "web_fetch", "tool_search_tool"}
)
_UNSUPPORTED_HOSTED_TOOL_PREFIXES = (
    "bash_",
    "text_editor_",
    "str_replace",
    "computer_",
    "memory_",
    "web_fetch",
    "tool_search",
)


def _is_unsupported_hosted_tool(tool_type: str, tool_name: str, tool_dict: Dict[str, Any]) -> bool:
    """A hosted tool is one declared by ``type`` (versioned) with no ``input_schema``."""
    if "input_schema" in tool_dict:
        return False
    if tool_name in _UNSUPPORTED_HOSTED_TOOL_NAMES:
        return True
    return any(tool_type.startswith(prefix) for prefix in _UNSUPPORTED_HOSTED_TOOL_PREFIXES)


def translate_mcp_servers_to_responses_api(mcp_servers: object) -> List[Dict[str, Any]]:
    """Translate Anthropic ``mcp_servers`` into OpenAI Responses ``mcp`` tools.

    Anthropic: {"type": "url", "url": ..., "name": ..., "authorization_token": ...,
                "tool_configuration": {"allowed_tools": [...]}}
    Responses: {"type": "mcp", "server_label": ..., "server_url": ..., "headers": {...},
                "allowed_tools": [...], "require_approval": "never"}
    """
    if not isinstance(mcp_servers, list):
        return []
    result: List[Dict[str, Any]] = []
    for server in mcp_servers:
        if not isinstance(server, dict):
            continue
        server_dict = cast(Dict[str, Any], server)
        url = server_dict.get("url")
        name = server_dict.get("name")
        if not isinstance(url, str) or not url or not isinstance(name, str) or not name:
            continue
        mcp_tool: Dict[str, Any] = {
            "type": "mcp",
            "server_label": name,
            "server_url": url,
            # Anthropic executes these without a client approval round-trip.
            "require_approval": "never",
        }
        token = server_dict.get("authorization_token")
        if isinstance(token, str) and token:
            mcp_tool["headers"] = {"Authorization": f"Bearer {token}"}
        tool_config = server_dict.get("tool_configuration")
        if isinstance(tool_config, dict):
            allowed = tool_config.get("allowed_tools")
            if isinstance(allowed, list) and allowed:
                mcp_tool["allowed_tools"] = [t for t in allowed if isinstance(t, str)]
        result.append(mcp_tool)
    return result


def _item_field(item: object, field: str) -> object:
    if isinstance(item, dict):
        return cast(Dict[str, object], item).get(field)
    return getattr(item, field, None)


def output_item_type(item: object) -> Optional[str]:
    value = _item_field(item, "type")
    return value if isinstance(value, str) else None


def build_mcp_tool_blocks(item: object) -> List[Dict[str, Any]]:
    """Translate a Responses ``mcp_call`` output item to Anthropic blocks.

    Emits a ``mcp_tool_use`` block plus a ``mcp_tool_result`` block, mirroring how
    Anthropic represents a server-executed MCP tool call in assistant content.
    """
    call_id_value = _item_field(item, "id")
    call_id = call_id_value if isinstance(call_id_value, str) else ""
    name_value = _item_field(item, "name")
    name = name_value if isinstance(name_value, str) else ""
    server_value = _item_field(item, "server_label")
    server_label = server_value if isinstance(server_value, str) else ""
    arguments = _item_field(item, "arguments")
    try:
        input_data = json.loads(arguments) if isinstance(arguments, str) and arguments else {}
    except (json.JSONDecodeError, TypeError):
        input_data = {}
    error = _item_field(item, "error")
    output = _item_field(item, "output")
    is_error = bool(error)
    if is_error:
        result_text = error if isinstance(error, str) else ""
    else:
        result_text = output if isinstance(output, str) else ""

    return [
        {
            "type": "mcp_tool_use",
            "id": call_id,
            "name": name,
            "server_name": server_label,
            "input": input_data,
        },
        {
            "type": "mcp_tool_result",
            "tool_use_id": call_id,
            "is_error": is_error,
            "content": [{"type": "text", "text": result_text}],
        },
    ]


class LiteLLMAnthropicToResponsesAPIAdapter:
    """
    Converts Anthropic /v1/messages requests to OpenAI Responses API format and
    converts Responses API responses back to Anthropic format.
    """

    # ------------------------------------------------------------------ #
    # Request translation: Anthropic -> Responses API                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _translate_anthropic_image_source_to_url(source: Dict[str, object]) -> Optional[str]:
        """Convert Anthropic image source to a URL string."""
        source_type = source.get("type")
        if source_type == "base64":
            media_type_value = source.get("media_type", "image/jpeg")
            media_type = media_type_value if isinstance(media_type_value, str) else "image/jpeg"
            data = source.get("data")
            return f"data:{media_type};base64,{data}" if isinstance(data, str) and data else None
        elif source_type == "url":
            url = source.get("url")
            return url if isinstance(url, str) else None
        return None

    def translate_messages_to_responses_input(
        self,
        messages: List[
            Union[
                AnthropicMessagesUserMessageParam,
                AnthopicMessagesAssistantMessageParam,
            ]
        ],
        *,
        drop_thinking_blocks: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Convert Anthropic messages list to Responses API `input` items.

        Mapping:
          user text          -> message(role=user, input_text)
          user image         -> message(role=user, input_image)
          user tool_result   -> function_call_output
          assistant text     -> message(role=assistant, output_text)
          assistant tool_use -> function_call
        """
        input_items: List[Dict[str, Any]] = []

        for m in messages:
            role = m["role"]
            content = m.get("content")

            if role == "user":
                if isinstance(content, str):
                    input_items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": content}],
                        }
                    )
                elif isinstance(content, list):
                    user_parts: List[Dict[str, Any]] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            user_parts.append({"type": "input_text", "text": block.get("text", "")})
                        elif btype == "image":
                            source = block.get("source")
                            source_dict = (
                                cast(Dict[str, object], source)  # cast-ok: guarded by the dict check
                                if isinstance(source, dict)
                                else {}
                            )
                            url = self._translate_anthropic_image_source_to_url(source_dict)
                            if url:
                                user_parts.append({"type": "input_image", "image_url": url})
                        elif btype == "tool_result":
                            if user_parts:
                                input_items.append(
                                    {
                                        "type": "message",
                                        "role": "user",
                                        "content": user_parts,
                                    }
                                )
                                user_parts = []
                            tool_use_id = block.get("tool_use_id", "")
                            inner = block.get("content")
                            if inner is None:
                                output: object = ""
                            elif isinstance(inner, str):
                                output = inner
                            elif isinstance(inner, list):
                                output_parts: List[Dict[str, object]] = []
                                for part in inner:
                                    if not isinstance(part, dict):
                                        continue
                                    part_dict = cast(Dict[str, object], part)  # cast-ok: guarded by the dict check
                                    if part_dict.get("type") == "text":
                                        text = part_dict.get("text")
                                        output_parts.append(
                                            {"type": "input_text", "text": text if isinstance(text, str) else ""}
                                        )
                                    elif part_dict.get("type") == "image":
                                        source = part_dict.get("source")
                                        source_dict = (
                                            cast(Dict[str, object], source)  # cast-ok: guarded by the dict check
                                            if isinstance(source, dict)
                                            else {}
                                        )
                                        url = self._translate_anthropic_image_source_to_url(source_dict)
                                        if url:
                                            output_parts.append({"type": "input_image", "image_url": url})
                                output = output_parts if output_parts else ""
                            else:
                                output = str(inner)
                            if block.get("is_error"):
                                # Preserve the error signal the model reads: Claude Code marks
                                # errored tool results with a <tool_use_error> wrapper, which
                                # the Responses function_call_output has no dedicated field for.
                                output = _wrap_tool_error(output)
                            # tool_result is a top-level item, not inside the message
                            input_items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": tool_use_id,
                                    "output": output,
                                }
                            )
                    if user_parts:
                        input_items.append(
                            {
                                "type": "message",
                                "role": "user",
                                "content": user_parts,
                            }
                        )

            elif role == "assistant":
                if isinstance(content, str):
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    )
                elif isinstance(content, list):
                    asst_parts: List[Dict[str, Any]] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            asst_parts.append({"type": "output_text", "text": block.get("text", "")})
                        elif btype == "tool_use":
                            if asst_parts:
                                input_items.append(
                                    {
                                        "type": "message",
                                        "role": "assistant",
                                        "content": asst_parts,
                                    }
                                )
                                asst_parts = []
                            # tool_use becomes a top-level function_call item
                            input_items.append(
                                {
                                    "type": "function_call",
                                    "call_id": block.get("id", ""),
                                    "name": truncate_tool_name(block.get("name", "")),
                                    "arguments": json.dumps(block.get("input", {})),
                                }
                            )
                        elif btype == "thinking":
                            if drop_thinking_blocks:
                                continue
                            thinking_text = block.get("thinking", "")
                            reasoning_state = decode_reasoning_signature(block.get("signature"))
                            if reasoning_state is not None:
                                if asst_parts:
                                    input_items.append(
                                        {
                                            "type": "message",
                                            "role": "assistant",
                                            "content": asst_parts,
                                        }
                                    )
                                    asst_parts = []
                                input_items.append(
                                    {
                                        "type": "reasoning",
                                        "id": reasoning_state.id,
                                        "encrypted_content": reasoning_state.encrypted_content,
                                        "summary": (
                                            [{"type": "summary_text", "text": thinking_text}]
                                            if isinstance(thinking_text, str) and thinking_text
                                            else []
                                        ),
                                    }
                                )
                            elif thinking_text:
                                asst_parts.append({"type": "output_text", "text": thinking_text})
                    if asst_parts:
                        input_items.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": asst_parts,
                            }
                        )

        return input_items

    def translate_tools_to_responses_api(
        self,
        tools: List[AllAnthropicToolsValues],
    ) -> List[Dict[str, Any]]:
        """Convert Anthropic tool definitions to Responses API tools.

        Client-side function tools map to Responses ``function`` tools. Anthropic
        hosted tools are mapped where the Responses API has an equivalent
        (``web_search`` -> ``web_search_preview``, ``code_execution`` ->
        ``code_interpreter``) and otherwise skipped, rather than mangled into a
        parameter-less function tool that the Responses API would reject.
        """
        result: List[Dict[str, Any]] = []
        for tool in tools:
            tool_dict = cast(Dict[str, Any], tool)  # cast-ok: validated by the Anthropic tool schema
            tool_type = tool_dict.get("type", "")
            tool_type_str = tool_type if isinstance(tool_type, str) else ""
            tool_name_value = tool_dict.get("name", "")
            tool_name = tool_name_value if isinstance(tool_name_value, str) else ""

            # web_search hosted tool -> Responses web_search_preview
            if tool_type_str.startswith("web_search") or tool_name == "web_search":
                result.append({"type": "web_search_preview"})
                continue
            # code_execution hosted tool -> Responses code_interpreter
            if tool_type_str.startswith("code_execution") or tool_name == "code_execution":
                result.append({"type": "code_interpreter", "container": {"type": "auto"}})
                continue
            # Other Anthropic hosted tools have no Responses equivalent: skip rather
            # than emit a parameter-less function tool the API would reject.
            if _is_unsupported_hosted_tool(tool_type_str, tool_name, tool_dict):
                continue

            func_tool: Dict[str, Any] = {"type": "function", "name": truncate_tool_name(tool_name)}
            if "description" in tool_dict:
                func_tool["description"] = tool_dict["description"]
            if "input_schema" in tool_dict:
                func_tool["parameters"] = sanitize_tool_parameters(tool_dict["input_schema"])
            if "strict" in tool_dict:
                func_tool["strict"] = tool_dict["strict"]
            result.append(func_tool)
        return result

    @staticmethod
    def translate_tool_choice_to_responses_api(
        tool_choice: AnthropicMessagesToolChoice,
    ) -> Union[str, Dict[str, str]]:
        """Convert Anthropic tool_choice to Responses API tool_choice."""
        tc_type = tool_choice.get("type")
        if tc_type == "any":
            return "required"
        elif tc_type == "tool":
            return {"type": "function", "name": truncate_tool_name(tool_choice.get("name", ""))}
        elif tc_type == "none":
            return "none"
        return "auto"

    @staticmethod
    def translate_context_management_to_responses_api(
        context_management: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Convert Anthropic context_management dict to OpenAI Responses API array format.

        Anthropic format: {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}
        OpenAI format:    [{"type": "compaction", "compact_threshold": 150000}]
        """
        if not isinstance(context_management, dict):
            return None

        edits = context_management.get("edits", [])
        if not isinstance(edits, list):
            return None

        result: List[Dict[str, Any]] = []
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            edit_type = edit.get("type", "")
            if edit_type == "compact_20260112":
                entry: Dict[str, Any] = {"type": "compaction"}
                trigger = edit.get("trigger")
                if isinstance(trigger, dict) and trigger.get("value") is not None:
                    entry["compact_threshold"] = int(trigger["value"])
                result.append(entry)

        return result if result else None

    @staticmethod
    def translate_thinking_to_reasoning(
        thinking: Dict[str, Any],
        output_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert Anthropic thinking param to Responses API reasoning param.

        ``thinking.budget_tokens`` is bucketed via the shared
        ``reasoning_effort_from_thinking_budget`` thresholds. For adaptive
        thinking, uses ``output_config.effort`` if available, otherwise defaults
        to medium.
        """
        if not isinstance(thinking, dict):
            return None

        thinking_type = thinking.get("type")

        if thinking_type == "adaptive":
            # Use output_config.effort if available
            effort = "medium"
            if isinstance(output_config, dict) and output_config.get("effort"):
                effort = output_config["effort"]
        elif thinking_type == "enabled":
            effort = reasoning_effort_from_thinking_budget(thinking.get("budget_tokens", 0))
        else:
            return None

        auto_summary = is_reasoning_auto_summary_enabled()
        result: Dict[str, Any] = {"effort": effort}
        summary = thinking.get("summary")
        if summary:
            result["summary"] = summary
        elif auto_summary:
            result["summary"] = "detailed"
        return result

    def translate_request(
        self,
        anthropic_request: AnthropicMessagesRequest,
    ) -> Dict[str, Any]:
        """
        Translate a full Anthropic /v1/messages request dict to
        litellm.responses() / litellm.aresponses() kwargs.
        """
        model: str = anthropic_request["model"]
        messages_list = cast(  # cast-ok: validated by AnthropicMessagesRequest
            List[
                Union[
                    AnthropicMessagesUserMessageParam,
                    AnthopicMessagesAssistantMessageParam,
                ]
            ],
            anthropic_request["messages"],
        )
        context_management = anthropic_request.get("context_management")
        clear_thinking = _context_management_clears_thinking(context_management)

        responses_kwargs: Dict[str, Any] = {
            "model": model,
            "input": self.translate_messages_to_responses_input(
                messages_list,
                drop_thinking_blocks=clear_thinking,
            ),
            "include": ["reasoning.encrypted_content"],
        }

        # system -> instructions
        system = _filter_billing_headers_from_system(anthropic_request.get("system"))
        if system:
            if isinstance(system, str):
                responses_kwargs["instructions"] = system
            elif isinstance(system, list):
                text_parts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"]
                responses_kwargs["instructions"] = "\n".join(filter(None, text_parts))

        # max_tokens -> max_output_tokens
        max_tokens = anthropic_request.get("max_tokens")
        if max_tokens is not None:
            responses_kwargs["max_output_tokens"] = max(OPENAI_MIN_RESPONSE_OUTPUT_TOKENS, max_tokens)

        # temperature / top_p passed through
        if "temperature" in anthropic_request:
            responses_kwargs["temperature"] = anthropic_request["temperature"]
        if "top_p" in anthropic_request:
            responses_kwargs["top_p"] = anthropic_request["top_p"]

        # tools (+ mcp_servers translated to Responses `mcp` tools)
        tools = anthropic_request.get("tools")
        translated_tools: List[Dict[str, Any]] = []
        if tools:
            translated_tools.extend(
                self.translate_tools_to_responses_api(
                    cast(List[AllAnthropicToolsValues], tools)  # cast-ok: validated by AnthropicMessagesRequest
                )
            )
        translated_tools.extend(translate_mcp_servers_to_responses_api(anthropic_request.get("mcp_servers")))
        if translated_tools:
            responses_kwargs["tools"] = translated_tools

        # tool_choice
        tool_choice = anthropic_request.get("tool_choice")
        if tool_choice:
            responses_kwargs["tool_choice"] = self.translate_tool_choice_to_responses_api(
                cast(AnthropicMessagesToolChoice, tool_choice)  # cast-ok: validated by AnthropicMessagesRequest
            )
            if "disable_parallel_tool_use" in tool_choice:
                responses_kwargs["parallel_tool_calls"] = not bool(tool_choice["disable_parallel_tool_use"])

        # thinking -> reasoning
        thinking = anthropic_request.get("thinking")
        if isinstance(thinking, dict):
            output_config = anthropic_request.get("output_config")
            reasoning = self.translate_thinking_to_reasoning(
                thinking,
                output_config=cast(  # cast-ok: validated by AnthropicMessagesRequest
                    Optional[Dict[str, Any]], output_config
                ),
            )
            if reasoning:
                responses_kwargs["reasoning"] = reasoning

        # output_format / output_config.format -> text format
        # output_format: {"type": "json_schema", "schema": {...}}
        # output_config: {"format": {"type": "json_schema", "schema": {...}}}
        output_format: Any = anthropic_request.get("output_format")
        output_config = anthropic_request.get("output_config")
        if not isinstance(output_format, dict) and isinstance(output_config, dict):
            output_format = output_config.get("format")  # type: ignore[assignment]
        if isinstance(output_format, dict) and output_format.get("type") == "json_schema":
            schema = output_format.get("schema")
            if schema:
                responses_kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "structured_output",
                        "schema": schema,
                        "strict": True,
                    }
                }

        # context_management: Anthropic dict -> OpenAI array
        if isinstance(context_management, dict):
            openai_cm = self.translate_context_management_to_responses_api(context_management)
            if openai_cm is not None:
                responses_kwargs["context_management"] = openai_cm

        # metadata user_id -> user
        metadata = anthropic_request.get("metadata")
        if isinstance(metadata, dict) and "user_id" in metadata:
            user_id = str(metadata["user_id"])
            responses_kwargs["user"] = user_id[:64]
            responses_kwargs["prompt_cache_key"] = hashlib.sha256(user_id.encode()).hexdigest()

        return responses_kwargs

    # ------------------------------------------------------------------ #
    # Response translation: Responses API -> Anthropic                    #
    # ------------------------------------------------------------------ #

    def translate_response(
        self,
        response: ResponsesAPIResponse,
        tool_name_mapping: Optional[Dict[str, str]] = None,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> AnthropicMessagesResponse:
        """
        Translate an OpenAI ResponsesAPIResponse to AnthropicMessagesResponse.
        """
        from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

        from litellm.types.llms.openai import ResponseAPIUsage

        content: List[Dict[str, Any]] = []
        stop_reason: AnthropicFinishReason = "end_turn"

        for item in response.output:
            reasoning_item = _parse_reasoning_output_item(item)
            if reasoning_item is not None:
                summary_text = "\n".join(summary.text for summary in reasoning_item.summary if summary.text)
                signature = encode_reasoning_signature(
                    reasoning_item.id,
                    reasoning_item.encrypted_content,
                )
                if summary_text or signature:
                    content.append(
                        AnthropicResponseContentBlockThinking(
                            type="thinking",
                            thinking=summary_text,
                            signature=signature,
                        ).model_dump()
                    )

            elif isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if getattr(part, "type", None) == "output_text":
                        content.append(
                            AnthropicResponseContentBlockText(type="text", text=getattr(part, "text", "")).model_dump()
                        )
                    elif getattr(part, "type", None) == "refusal":
                        content.append(
                            AnthropicResponseContentBlockText(
                                type="text",
                                text=getattr(part, "refusal", ""),
                            ).model_dump()
                        )
                        stop_reason = "refusal"

            elif isinstance(item, ResponseFunctionToolCall):
                try:
                    input_data = json.loads(item.arguments) if item.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    input_data = {}
                content.append(
                    AnthropicResponseContentBlockToolUse(
                        type="tool_use",
                        id=item.call_id or item.id or "",
                        name=(tool_name_mapping or {}).get(item.name, item.name),
                        input=input_data,
                    ).model_dump()
                )
                stop_reason = "tool_use"

            elif output_item_type(item) == "mcp_call":
                # Server-executed MCP tool call: emitted as content, turn continues.
                content.extend(build_mcp_tool_blocks(item))

            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "message":
                    for part in item.get("content", []):
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            content.append(
                                AnthropicResponseContentBlockText(type="text", text=part.get("text", "")).model_dump()
                            )
                        elif isinstance(part, dict) and part.get("type") == "refusal":
                            content.append(
                                AnthropicResponseContentBlockText(
                                    type="text",
                                    text=part.get("refusal", ""),
                                ).model_dump()
                            )
                            stop_reason = "refusal"
                elif item_type == "function_call":
                    try:
                        input_data = json.loads(item.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        input_data = {}
                    item_name_value = item.get("name", "")
                    item_name = item_name_value if isinstance(item_name_value, str) else ""
                    content.append(
                        AnthropicResponseContentBlockToolUse(
                            type="tool_use",
                            id=item.get("call_id") or item.get("id", ""),
                            name=(tool_name_mapping or {}).get(item_name, item_name),
                            input=input_data,
                        ).model_dump()
                    )
                    stop_reason = "tool_use"

        # status -> stop_reason override
        if response.status == "incomplete":
            incomplete_details = getattr(response, "incomplete_details", None)
            if isinstance(incomplete_details, dict):
                incomplete_reason = incomplete_details.get("reason")
            else:
                incomplete_reason = getattr(incomplete_details, "reason", None)
            if incomplete_reason in ("content_filter", "refusal"):
                stop_reason = "refusal"
            elif incomplete_reason in ("model_context_window_exceeded", "context_length_exceeded"):
                # The Responses API surfaces input overflow as a 4xx today; this branch
                # keeps parity if a provider ever reports it on a 200 response instead.
                stop_reason = "model_context_window_exceeded"
            else:
                stop_reason = "max_tokens"

        # stop_sequences: emulated by scanning output text (no native Responses param).
        matched_stop: Optional[str] = None
        normalized_stops = normalize_stop_sequences(stop_sequences)
        if normalized_stops and stop_reason != "refusal":
            matched_stop = apply_stop_sequences_to_content(content, normalized_stops)
            if matched_stop is not None:
                stop_reason = "stop_sequence"

        # usage
        raw_usage: Optional[ResponseAPIUsage] = response.usage
        total_input_tokens = int(getattr(raw_usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(raw_usage, "output_tokens", 0) or 0)
        input_token_details = getattr(raw_usage, "input_tokens_details", None)
        if isinstance(input_token_details, dict):
            details = cast(Dict[str, object], input_token_details)  # cast-ok: guarded by the dict check
            cache_read_input_tokens = _token_count(details.get("cached_tokens"))
            cache_creation_input_tokens = _token_count(
                details.get("cache_write_tokens", 0) or details.get("cache_creation_tokens", 0)
            )
        else:
            cache_read_value: object = getattr(input_token_details, "cached_tokens", 0)
            cache_creation_value: object = getattr(input_token_details, "cache_write_tokens", 0) or getattr(
                input_token_details, "cache_creation_tokens", 0
            )
            cache_read_input_tokens = _token_count(cache_read_value)
            cache_creation_input_tokens = _token_count(cache_creation_value)
        input_tokens = max(0, total_input_tokens - cache_read_input_tokens - cache_creation_input_tokens)

        anthropic_usage = AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
        service_tier = map_service_tier(getattr(response, "service_tier", None))
        if service_tier is not None:
            anthropic_usage["service_tier"] = service_tier

        return AnthropicMessagesResponse(
            id=response.id,
            type="message",
            role="assistant",
            model=response.model or "unknown-model",
            stop_sequence=matched_stop,
            usage=anthropic_usage,  # type: ignore
            content=content,  # type: ignore
            stop_reason=stop_reason,
        )
