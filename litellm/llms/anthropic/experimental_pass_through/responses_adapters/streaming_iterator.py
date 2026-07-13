# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
from collections import deque
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import TypeAdapter

from litellm._uuid import uuid

from .transformation import (
    build_mcp_tool_blocks,
    encode_reasoning_item_signature,
    find_earliest_stop_sequence,
    map_service_tier,
    normalize_stop_sequences,
    partial_stop_suffix_len,
)

_OBJECT_DICT_ADAPTER = TypeAdapter(Dict[str, object])


def _get_field(value: object, field: str) -> object | None:
    if isinstance(value, dict):
        return _OBJECT_DICT_ADAPTER.validate_python(value).get(field)
    attribute = getattr(value, field, None)
    return attribute if isinstance(attribute, object) else None


def _token_count(value: object | None) -> int:
    return value if isinstance(value, int) else 0


class AnthropicResponsesStreamWrapper:
    """
    Wraps a Responses API streaming iterator and re-emits events in Anthropic SSE format.

    Responses API event flow (relevant subset):
      response.created                   -> message_start
      response.output_item.added         -> content_block_start (if message/function_call)
      response.output_text.delta         -> content_block_delta (text_delta)
      response.reasoning_summary_text.delta -> content_block_delta (thinking_delta)
      response.function_call_arguments.delta -> content_block_delta (input_json_delta)
      response.output_item.done          -> content_block_stop
      response.completed                 -> message_delta + message_stop
    """

    def __init__(
        self,
        responses_stream: Any,
        model: str,
        prompt_tokens: Optional[int] = None,
        tool_name_mapping: Optional[Dict[str, str]] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> None:
        self.responses_stream = responses_stream
        self.model = model
        self._message_id: str = f"msg_{uuid.uuid4()}"
        self._current_block_index: int = -1
        # Map item_id -> content_block_index so we can stop the right block later
        self._item_id_to_block_index: Dict[str, int] = {}
        # Track open function_call items by item_id so we can emit tool_use start
        self._pending_tool_ids: Dict[str, str] = {}  # item_id -> call_id / name accumulator
        self._sent_message_start = False
        self._sent_message_stop = False
        self._prompt_tokens = prompt_tokens
        self._tool_name_mapping = tool_name_mapping or {}
        self._chunk_queue: deque[Dict[str, Any]] = deque()
        # stop_sequences are emulated (no native Responses param): buffer text so a
        # sequence spanning multiple deltas is caught, then truncate and stop.
        self._stop_sequences = normalize_stop_sequences(stop_sequences)
        self._stopped_by_sequence: Optional[str] = None
        self._text_hold: Dict[int, str] = {}  # block_index -> withheld tail

    def _make_message_start(self) -> Dict[str, Any]:
        initial_usage = {
            "input_tokens": self._prompt_tokens or 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        return {
            "type": "message_start",
            "message": {
                "id": self._message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": initial_usage,
            },
        }

    def _next_block_index(self) -> int:
        self._current_block_index += 1
        return self._current_block_index

    def _emit_mcp_call_blocks(self, item: object) -> None:
        """Emit a server-executed MCP call as atomic mcp_tool_use + mcp_tool_result blocks."""
        for block in build_mcp_tool_blocks(item):
            block_idx = self._next_block_index()
            self._chunk_queue.append(
                {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": block,
                }
            )
            self._chunk_queue.append({"type": "content_block_stop", "index": block_idx})

    def _queue_text_delta(self, block_idx: int, text: str) -> None:
        if text:
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "text_delta", "text": text},
                }
            )

    def _emit_text_with_stop_check(self, block_idx: int, item_id: Optional[str], delta: str) -> None:
        """Emit text while emulating stop sequences.

        Withholds any trailing chars that could still grow into a stop sequence; on a
        full match, emits the prefix, closes the block, and records the stop so the rest
        of the stream is suppressed.
        """
        combined = self._text_hold.pop(block_idx, "") + delta
        match = find_earliest_stop_sequence(combined, self._stop_sequences)
        if match is not None:
            idx, seq = match
            self._queue_text_delta(block_idx, combined[:idx])
            self._stopped_by_sequence = seq
            # Drop this block from the id map so its later output_item.done is a no-op.
            if item_id:
                self._item_id_to_block_index.pop(item_id, None)
            self._chunk_queue.append({"type": "content_block_stop", "index": block_idx})
            return
        hold_len = partial_stop_suffix_len(combined, self._stop_sequences)
        if hold_len:
            self._text_hold[block_idx] = combined[len(combined) - hold_len :]
            self._queue_text_delta(block_idx, combined[: len(combined) - hold_len])
        else:
            self._queue_text_delta(block_idx, combined)

    def _queue_error(self, message: str, request_id: Optional[str] = None) -> None:
        error_chunk: Dict[str, Any] = {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        }
        if request_id:
            error_chunk["request_id"] = request_id
        self._chunk_queue.append(error_chunk)
        self._sent_message_stop = True

    def _process_event(self, event: Any) -> None:
        """Convert one Responses API event into zero or more Anthropic chunks queued for emission."""
        event_type = getattr(event, "type", None)
        if event_type is None and isinstance(event, dict):
            event_type = event.get("type")

        if event_type is None:
            return

        if self._sent_message_stop:
            return

        if event_type == "error":
            error_obj = _get_field(event, "error")
            message = _get_field(event, "message") or _get_field(error_obj, "message") or "Upstream response failed"
            self._queue_error(message=str(message))
            return

        if event_type == "response.failed":
            response_obj = _get_field(event, "response")
            error_obj = _get_field(response_obj, "error")
            message = _get_field(error_obj, "message") or "Upstream response failed"
            request_id = _get_field(response_obj, "id")
            self._queue_error(
                message=str(message),
                request_id=str(request_id) if request_id else None,
            )
            return

        # Once a stop sequence has been hit, suppress all further content and wait for
        # the terminal event (Anthropic stops emitting at the sequence).
        if self._stopped_by_sequence is not None and event_type not in (
            "response.completed",
            "response.incomplete",
        ):
            return

        # ---- message_start ----
        if event_type == "response.created":
            if not self._sent_message_start:
                self._sent_message_start = True
                self._chunk_queue.append(self._make_message_start())
            return

        # ---- content_block_start for a new output message item ----
        if event_type == "response.output_item.added":
            item = getattr(event, "item", None) or (event.get("item") if isinstance(event, dict) else None)
            if item is None:
                return
            item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
            item_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)

            if item_type == "message":
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            elif item_type == "function_call":
                call_id = (
                    getattr(item, "call_id", None) or (item.get("call_id") if isinstance(item, dict) else None) or ""
                )
                name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None) or ""
                name = self._tool_name_mapping.get(name, name)
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                    self._pending_tool_ids[item_id] = call_id
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": {},
                        },
                    }
                )
            elif item_type == "reasoning":
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                    }
                )
            return

        # ---- text delta ----
        if event_type == "response.output_text.delta":
            item_id = getattr(event, "item_id", None) or (event.get("item_id") if isinstance(event, dict) else None)
            delta = getattr(event, "delta", "") or (event.get("delta", "") if isinstance(event, dict) else "")
            block_idx = self._item_id_to_block_index.get(item_id, -1) if item_id else self._current_block_index
            if block_idx < 0:
                # Some providers (e.g. LMStudio) skip response.output_item.added,
                # so no text block is open yet; synthesize content_block_start
                # instead of emitting a delta with index -1
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            if not self._stop_sequences:
                self._queue_text_delta(block_idx, delta)
            else:
                self._emit_text_with_stop_check(block_idx, item_id, delta)
            return

        if event_type == "response.refusal.delta":
            item_id = getattr(event, "item_id", None) or (event.get("item_id") if isinstance(event, dict) else None)
            delta = getattr(event, "delta", "") or (event.get("delta", "") if isinstance(event, dict) else "")
            block_idx = self._item_id_to_block_index.get(item_id, -1) if item_id else self._current_block_index
            if block_idx >= 0:
                self._chunk_queue.append(
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": delta},
                    }
                )
            return

        # ---- reasoning summary text delta ----
        if event_type == "response.reasoning_summary_text.delta":
            item_id = getattr(event, "item_id", None) or (event.get("item_id") if isinstance(event, dict) else None)
            delta = getattr(event, "delta", "") or (event.get("delta", "") if isinstance(event, dict) else "")
            block_idx = (
                self._item_id_to_block_index.get(item_id, self._current_block_index)
                if item_id
                else self._current_block_index
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "thinking_delta", "thinking": delta},
                }
            )
            return

        # ---- function call arguments delta ----
        if event_type == "response.function_call_arguments.delta":
            item_id = getattr(event, "item_id", None) or (event.get("item_id") if isinstance(event, dict) else None)
            delta = getattr(event, "delta", "") or (event.get("delta", "") if isinstance(event, dict) else "")
            block_idx = (
                self._item_id_to_block_index.get(item_id, self._current_block_index)
                if item_id
                else self._current_block_index
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                }
            )
            return

        # ---- output item done -> content_block_stop ----
        if event_type == "response.output_item.done":
            item = getattr(event, "item", None) or (event.get("item") if isinstance(event, dict) else None)
            item_id = (
                getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None) if item else None
            )
            item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
            if item_type == "mcp_call":
                # Server-executed MCP call arrives complete: emit its blocks atomically.
                self._emit_mcp_call_blocks(item)
                return
            if item_id:
                block_idx = self._item_id_to_block_index.pop(item_id, None)
                if block_idx is None:
                    return
            elif item_type in ("message", "function_call", "reasoning") and self._current_block_index >= 0:
                block_idx = self._current_block_index
            else:
                return
            # Flush any text withheld for stop-sequence detection now that the block is done.
            held = self._text_hold.pop(block_idx, "")
            if held:
                self._queue_text_delta(block_idx, held)
            signature = encode_reasoning_item_signature(item)
            if signature is not None:
                self._chunk_queue.append(
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {
                            "type": "signature_delta",
                            "signature": signature,
                        },
                    }
                )
            self._chunk_queue.append({"type": "content_block_stop", "index": block_idx})
            return

        # ---- response completed -> message_delta + message_stop ----
        if event_type in (
            "response.completed",
            "response.incomplete",
        ):
            response_obj = _get_field(event, "response")
            stop_reason = "end_turn"
            input_tokens = 0
            output_tokens = 0
            cache_creation_tokens = 0
            cache_read_tokens = 0

            service_tier: Optional[str] = None
            if response_obj is not None:
                status = _get_field(response_obj, "status")
                if status == "incomplete":
                    stop_reason = "max_tokens"
                    incomplete_details = _get_field(response_obj, "incomplete_details")
                    incomplete_reason = _get_field(incomplete_details, "reason")
                    if incomplete_reason in ("content_filter", "refusal"):
                        stop_reason = "refusal"
                    elif incomplete_reason in ("model_context_window_exceeded", "context_length_exceeded"):
                        stop_reason = "model_context_window_exceeded"
                service_tier = map_service_tier(_get_field(response_obj, "service_tier"))
                usage = _get_field(response_obj, "usage")
                if usage is not None:
                    total_input_tokens = _token_count(_get_field(usage, "input_tokens"))
                    output_tokens = _token_count(_get_field(usage, "output_tokens"))
                    input_details = _get_field(usage, "input_tokens_details")
                    cache_creation_tokens = _token_count(
                        _get_field(usage, "cache_creation_input_tokens")
                        or _get_field(usage, "cache_creation_tokens")
                        or _get_field(input_details, "cache_write_tokens")
                        or _get_field(input_details, "cache_creation_tokens")
                    )
                    cache_read_tokens = _token_count(
                        _get_field(usage, "cache_read_input_tokens") or _get_field(input_details, "cached_tokens")
                    )
                    input_tokens = max(0, total_input_tokens - cache_creation_tokens - cache_read_tokens)

            # Check if tool_use was in the output to override stop_reason
            if response_obj is not None and event_type == "response.completed":
                output = _get_field(response_obj, "output")
                output = output if isinstance(output, (list, tuple)) else ()
                for out_item in output:
                    out_type = _get_field(out_item, "type")
                    if out_type == "function_call":
                        stop_reason = "tool_use"
                        break
                    content = _get_field(out_item, "content")
                    content = content if isinstance(content, (list, tuple)) else ()
                    if any(_get_field(part, "type") == "refusal" for part in content):
                        stop_reason = "refusal"

            # An emulated stop sequence takes precedence over the natural stop reason.
            if self._stopped_by_sequence is not None:
                stop_reason = "stop_sequence"

            usage_delta: Dict[str, Any] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if cache_creation_tokens:
                usage_delta["cache_creation_input_tokens"] = cache_creation_tokens
            if cache_read_tokens:
                usage_delta["cache_read_input_tokens"] = cache_read_tokens
            if service_tier is not None:
                usage_delta["service_tier"] = service_tier

            self._chunk_queue.append(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": self._stopped_by_sequence},
                    "usage": usage_delta,
                }
            )
            self._chunk_queue.append({"type": "message_stop"})
            self._sent_message_stop = True
            return

    def __aiter__(self) -> "AnthropicResponsesStreamWrapper":
        return self

    async def __anext__(self) -> Dict[str, Any]:
        # Return any queued chunks first
        if self._chunk_queue:
            return self._chunk_queue.popleft()

        if self._sent_message_stop:
            raise StopAsyncIteration

        # Emit message_start if not yet done (fallback if response.created wasn't fired)
        if not self._sent_message_start:
            self._sent_message_start = True
            self._chunk_queue.append(self._make_message_start())
            return self._chunk_queue.popleft()

        # Consume the upstream stream
        try:
            async for event in self.responses_stream:
                self._process_event(event)
                if self._chunk_queue:
                    return self._chunk_queue.popleft()
        except StopAsyncIteration:
            pass

        # Drain any remaining queued chunks
        if self._chunk_queue:
            return self._chunk_queue.popleft()

        self._queue_error("Upstream response ended before a terminal event")
        return self._chunk_queue.popleft()

    async def async_anthropic_sse_wrapper(self) -> AsyncIterator[bytes]:
        """Yield SSE-encoded bytes for each Anthropic event chunk."""
        async for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                yield chunk
