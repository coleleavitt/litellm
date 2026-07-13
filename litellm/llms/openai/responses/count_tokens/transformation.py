"""
OpenAI Responses API token counting transformation logic.

This module handles the transformation of requests to OpenAI's /v1/responses/input_tokens endpoint.
"""

from itertools import groupby
from typing import Any, Dict, List, Optional, Union, cast

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)
from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthopicMessagesAssistantMessageParam,
    AnthropicMessagesUserMessageParam,
)


_ANTHROPIC_ADAPTER = LiteLLMAnthropicToResponsesAPIAdapter()


class OpenAICountTokensConfig:
    """
    Configuration and transformation logic for OpenAI Responses API token counting.

    OpenAI Responses API Token Counting Specification:
    - Endpoint: POST https://api.openai.com/v1/responses/input_tokens
    - Response: {"input_tokens": <number>}
    """

    def get_openai_count_tokens_endpoint(self, api_base: Optional[str] = None) -> str:
        base = api_base or "https://api.openai.com/v1"
        base = base.rstrip("/")
        return f"{base}/responses/input_tokens"

    def transform_request_to_count_tokens(
        self,
        model: str,
        input: Union[str, List[Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transform request to OpenAI Responses API token counting format.

        The Responses API uses `input` (not `messages`) and `instructions` (not `system`).
        """
        request: Dict[str, Any] = {
            "model": model,
            "input": input,
        }

        if instructions is not None:
            request["instructions"] = instructions

        if tools is not None:
            request["tools"] = self._transform_tools_for_responses_api(tools)

        return request

    def get_required_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def validate_request(self, model: str, input: Union[str, List[Any]]) -> None:
        if not model:
            raise ValueError("model parameter is required")

        if not input:
            raise ValueError("input parameter is required")

    @staticmethod
    def _transform_tools_for_responses_api(
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Transform OpenAI chat tools format to Responses API tools format.

        Chat format:  {"type": "function", "function": {"name": "...", "parameters": {...}}}
        Responses format: {"type": "function", "name": "...", "parameters": {...}}
        """
        transformed = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                item: Dict[str, Any] = {
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }
                if "strict" in func:
                    item["strict"] = func["strict"]
                transformed.append(item)
            elif OpenAICountTokensConfig._is_anthropic_tool(tool):
                transformed.extend(
                    _ANTHROPIC_ADAPTER.translate_tools_to_responses_api(cast(List[AllAnthropicToolsValues], [tool]))
                )
            else:
                # Pass through non-function tools (e.g., web_search, file_search)
                transformed.append(tool)
        return transformed

    @staticmethod
    def _is_anthropic_tool(tool: Dict[str, Any]) -> bool:
        tool_type = tool.get("type")
        return "input_schema" in tool or (isinstance(tool_type, str) and tool_type.startswith("web_search"))

    @staticmethod
    def _is_top_level_anthropic_block(role: str, block: object) -> bool:
        if not isinstance(block, dict):
            return False
        block_type = block.get("type")
        return (role == "user" and block_type == "tool_result") or (role == "assistant" and block_type == "tool_use")

    @classmethod
    def _split_anthropic_message(cls, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = message.get("content")
        if not isinstance(content, list):
            return [message]
        role = str(message.get("role", ""))
        return [
            {**message, "content": list(blocks)}
            for _, blocks in groupby(
                content,
                key=lambda block: cls._is_top_level_anthropic_block(role, block),
            )
        ]

    @classmethod
    def _anthropic_messages_to_responses_input(
        cls,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        split_messages = [segment for message in messages for segment in cls._split_anthropic_message(message)]
        translated = _ANTHROPIC_ADAPTER.translate_messages_to_responses_input(
            cast(
                List[Union[AnthropicMessagesUserMessageParam, AnthopicMessagesAssistantMessageParam]],
                split_messages,
            )
        )
        return [cls._normalize_anthropic_input_item(item) for item in translated]

    @staticmethod
    def _normalize_anthropic_input_item(item: Dict[str, Any]) -> Dict[str, Any]:
        if item.get("type") != "message" or not isinstance(item.get("content"), list):
            return item
        content = [
            {**part, "type": "input_text"}
            if isinstance(part, dict) and part.get("type") == "output_text"
            else {**part, "detail": part.get("detail", "auto")}
            if isinstance(part, dict) and part.get("type") == "input_image"
            else part
            for part in item["content"]
        ]
        return {**item, "content": content}

    @staticmethod
    def _anthropic_system_to_instructions(system: Optional[object]) -> Optional[str]:
        if system is None or isinstance(system, str):
            return system
        if isinstance(system, list):
            instructions = "\n".join(
                str(block.get("text", ""))
                for block in system
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
            )
            return instructions or None
        return str(system)

    @classmethod
    def _is_anthropic_request(
        cls,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        system: Optional[object],
    ) -> bool:
        if isinstance(system, list) or any(cls._is_anthropic_tool(tool) for tool in tools or []):
            return True
        return any(
            isinstance(block, dict)
            and (
                block.get("type") in ("tool_use", "tool_result", "thinking", "redacted_thinking", "document")
                or (block.get("type") == "image" and "source" in block)
            )
            for message in messages
            for block in (message.get("content") if isinstance(message.get("content"), list) else [])
        )

    @classmethod
    def messages_to_responses_input(
        cls,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[object] = None,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Convert standard chat messages format to OpenAI Responses API input format.

        Returns:
            (input_items, instructions) tuple where instructions is extracted
            from system/developer messages.
        """
        if cls._is_anthropic_request(messages=messages, tools=tools, system=system):
            return cls._anthropic_messages_to_responses_input(messages), cls._anthropic_system_to_instructions(system)

        input_items: List[Dict[str, Any]] = []
        instructions_parts: List[str] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""

            if role in ("system", "developer"):
                # Extract system/developer messages as instructions
                if isinstance(content, str):
                    instructions_parts.append(content)
                elif isinstance(content, list):
                    # Handle content blocks - extract text
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    instructions_parts.append("\n".join(text_parts))
            elif role == "user":
                if isinstance(content, list):
                    # Extract text from content blocks for Responses API
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = "\n".join(text_parts)
                input_items.append({"role": "user", "content": content})
            elif role == "assistant":
                # Map tool_calls to Responses API function_call items
                tool_calls = msg.get("tool_calls")
                if content:
                    input_items.append({"role": "assistant", "content": content})
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": tc.get("id", ""),
                                "name": func.get("name", ""),
                                "arguments": func.get("arguments", ""),
                            }
                        )
                elif not content:
                    input_items.append({"role": "assistant", "content": content})
            elif role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": content if isinstance(content, str) else str(content),
                    }
                )

        instructions = "\n".join(instructions_parts) if instructions_parts else None
        if instructions is None and system is not None:
            instructions = system if isinstance(system, str) else str(system)
        return input_items, instructions
