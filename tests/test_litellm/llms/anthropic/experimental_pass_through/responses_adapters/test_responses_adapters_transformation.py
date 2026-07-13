"""
Tests for LiteLLMAnthropicToResponsesAPIAdapter
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/transformation.py)
"""

import hashlib
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../../../../.."))

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    truncate_tool_name,
)
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
    encode_reasoning_signature,
)
from litellm.types.llms.anthropic import AnthropicMessagesRequest


def _make_request(**overrides) -> AnthropicMessagesRequest:
    base: dict = {
        "model": "openai.gpt-5.1-codex",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1024,
    }
    base.update(overrides)
    return AnthropicMessagesRequest(**base)


_ADAPTER = LiteLLMAnthropicToResponsesAPIAdapter()


# ---------------------------------------------------------------------------
# context_management conversion
# ---------------------------------------------------------------------------


class TestContextManagementConversion:
    """Anthropic dict -> OpenAI array conversion for context_management."""

    def test_compact_edit_converted_to_array(self):
        """compact_20260112 with trigger maps to OpenAI compaction entry."""
        cm = {
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 150000},
                }
            ]
        }
        result = _ADAPTER.translate_context_management_to_responses_api(cm)
        assert result == [{"type": "compaction", "compact_threshold": 150000}]

    def test_compact_edit_without_trigger(self):
        """compact_20260112 without a trigger still maps to a compaction entry."""
        cm = {"edits": [{"type": "compact_20260112"}]}
        result = _ADAPTER.translate_context_management_to_responses_api(cm)
        assert result == [{"type": "compaction"}]

    def test_unknown_edit_type_is_dropped(self):
        """Anthropic-only edit types (e.g. clear_thinking) are silently dropped."""
        cm = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
        result = _ADAPTER.translate_context_management_to_responses_api(cm)
        assert result is None

    def test_mixed_edits_only_known_types_kept(self):
        """Only compact_20260112 is converted; unknown types are dropped."""
        cm = {
            "edits": [
                {"type": "clear_thinking_20251015", "keep": "all"},
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 200000},
                },
            ]
        }
        result = _ADAPTER.translate_context_management_to_responses_api(cm)
        assert result == [{"type": "compaction", "compact_threshold": 200000}]

    def test_non_dict_returns_none(self):
        result = _ADAPTER.translate_context_management_to_responses_api([])  # type: ignore
        assert result is None

    def test_translate_request_includes_context_management(self):
        """translate_request converts context_management and sets it on kwargs."""
        req = _make_request(
            context_management={
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": 100000},
                    }
                ]
            }
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["context_management"] == [{"type": "compaction", "compact_threshold": 100000}]

    def test_translate_request_drops_anthropic_only_context_management(self):
        """context_management with only unknown edit types is omitted from kwargs."""
        req = _make_request(context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]})
        kwargs = _ADAPTER.translate_request(req)
        assert "context_management" not in kwargs

    def test_clear_thinking_drops_replayed_reasoning_and_fallback(self):
        reasoning = _make_reasoning_item(
            ["Reasoning to clear."],
            item_id="rs_clear",
            encrypted_content="encrypted-clear-state",
        )
        signed_block = _ADAPTER.translate_response(_make_mock_response(output=[reasoning]))["content"][0]
        req = _make_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        signed_block,
                        {
                            "type": "thinking",
                            "thinking": "Native reasoning to clear.",
                            "signature": "native-signature",
                        },
                    ],
                }
            ],
            context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "none"}]},
        )

        kwargs = _ADAPTER.translate_request(req)

        assert kwargs["input"] == []

    def test_clear_thinking_keep_all_preserves_replayed_reasoning(self):
        reasoning = _make_reasoning_item(
            ["Reasoning to retain."],
            item_id="rs_keep",
            encrypted_content="encrypted-keep-state",
        )
        signed_block = _ADAPTER.translate_response(_make_mock_response(output=[reasoning]))["content"][0]
        req = _make_request(
            messages=[{"role": "assistant", "content": [signed_block]}],
            context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        )

        kwargs = _ADAPTER.translate_request(req)

        assert kwargs["input"][0]["type"] == "reasoning"
        assert kwargs["input"][0]["encrypted_content"] == "encrypted-keep-state"


# ---------------------------------------------------------------------------
# structured output via output_config
# ---------------------------------------------------------------------------


class TestOutputConfigStructuredOutput:
    """output_config.format.json_schema -> OpenAI text.format conversion."""

    _SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
        },
        "required": ["name", "email"],
        "additionalProperties": False,
    }

    def test_output_config_format_json_schema_converted(self):
        """output_config.format.json_schema is converted to OpenAI text.format."""
        req = _make_request(output_config={"format": {"type": "json_schema", "schema": self._SCHEMA}})
        kwargs = _ADAPTER.translate_request(req)
        assert "text" in kwargs
        fmt = kwargs["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"] == self._SCHEMA
        assert fmt["strict"] is True
        assert fmt["name"] == "structured_output"

    def test_output_config_without_format_does_not_set_text(self):
        """output_config with only non-format keys doesn't produce text.format."""
        req = _make_request(output_config={"effort": "high"})
        kwargs = _ADAPTER.translate_request(req)
        assert "text" not in kwargs

    def test_output_format_still_works(self):
        """The original output_format field still takes precedence when present."""
        req = _make_request(output_format={"type": "json_schema", "schema": self._SCHEMA})
        kwargs = _ADAPTER.translate_request(req)
        assert "text" in kwargs
        assert kwargs["text"]["format"]["type"] == "json_schema"

    def test_output_format_takes_precedence_over_output_config(self):
        """output_format takes precedence over output_config.format."""
        other_schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
        req = _make_request(
            output_format={"type": "json_schema", "schema": self._SCHEMA},
            output_config={"format": {"type": "json_schema", "schema": other_schema}},
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["text"]["format"]["schema"] == self._SCHEMA


# ---------------------------------------------------------------------------
# translate_messages_to_responses_input
# ---------------------------------------------------------------------------


# Helper: cast plain dicts to the expected type so call sites stay clean.
def _translate_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    return _ADAPTER.translate_messages_to_responses_input(messages)  # type: ignore[arg-type]


class TestTranslateMessagesToResponsesInput:
    """Anthropic messages list -> OpenAI Responses API input items."""

    def test_user_string_content(self):
        """Plain string user message becomes a message with input_text."""
        messages = [{"role": "user", "content": "Hello world"}]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello world"}],
            }
        ]

    def test_user_list_text_block(self):
        """User message with text content block maps to input_text."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is 2+2?"}],
            }
        ]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What is 2+2?"}],
            }
        ]

    def test_user_multiple_text_blocks(self):
        """Multiple text blocks in a user message are all converted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                ],
            }
        ]
        result = _translate_messages(messages)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "input_text", "text": "First part."},
            {"type": "input_text", "text": "Second part."},
        ]

    def test_user_base64_image(self):
        """User message with base64 image source becomes input_image with data URL."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc123",
                        },
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert len(result) == 1
        assert result[0]["content"] == [{"type": "input_image", "image_url": "data:image/png;base64,abc123"}]

    def test_user_url_image(self):
        """User message with URL image source becomes input_image with the URL."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.com/img.jpg"},
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result[0]["content"] == [{"type": "input_image", "image_url": "https://example.com/img.jpg"}]

    def test_user_base64_image_empty_data_skipped(self):
        """Base64 image with empty data is skipped (no URL can be formed)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "",
                        },
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        # No user_parts -> no message item appended
        assert result == []

    def test_user_tool_result_string_content(self):
        """tool_result with string content becomes function_call_output."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_abc",
                        "content": "42 degrees",
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "42 degrees",
            }
        ]

    def test_user_tool_result_list_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_xyz",
                        "content": [
                            {"type": "text", "text": "Line 1"},
                            {"type": "text", "text": "Line 2"},
                        ],
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result[0]["output"] == [
            {"type": "input_text", "text": "Line 1"},
            {"type": "input_text", "text": "Line 2"},
        ]

    def test_user_tool_result_preserves_text_and_image_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_media",
                        "content": [
                            {"type": "text", "text": "Screenshot"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "abc123",
                                },
                            },
                        ],
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result[0]["output"] == [
            {"type": "input_text", "text": "Screenshot"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,abc123",
            },
        ]

    def test_user_tool_result_null_content(self):
        """tool_result with null content becomes empty string output."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_null", "content": None}],
            }
        ]
        result = _translate_messages(messages)
        assert result[0]["output"] == ""

    def test_assistant_string_content(self):
        """Plain string assistant message becomes a message with output_text."""
        messages = [{"role": "assistant", "content": "I can help with that."}]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I can help with that."}],
            }
        ]

    def test_assistant_text_block(self):
        """Assistant message with text block maps to output_text."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Here is the answer."}],
            }
        ]
        result = _translate_messages(messages)
        assert result[0]["content"] == [{"type": "output_text", "text": "Here is the answer."}]

    def test_assistant_tool_use_becomes_function_call(self):
        """Assistant tool_use block becomes a top-level function_call item."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "get_weather",
                        "input": {"location": "Boston"},
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "function_call",
                "call_id": "toolu_01",
                "name": "get_weather",
                "arguments": json.dumps({"location": "Boston"}),
            }
        ]

    def test_assistant_text_and_tool_blocks_preserve_order(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Before"},
                    {
                        "type": "tool_use",
                        "id": "toolu_order",
                        "name": "lookup",
                        "input": {"value": 1},
                    },
                    {"type": "text", "text": "After"},
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Before"}],
            },
            {
                "type": "function_call",
                "call_id": "toolu_order",
                "name": "lookup",
                "arguments": '{"value": 1}',
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "After"}],
            },
        ]

    def test_user_text_and_tool_result_blocks_preserve_order(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Before"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_order",
                        "content": "result",
                    },
                    {"type": "text", "text": "After"},
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Before"}],
            },
            {
                "type": "function_call_output",
                "call_id": "toolu_order",
                "output": "result",
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "After"}],
            },
        ]

    def test_assistant_thinking_block_becomes_output_text(self):
        """Assistant thinking block text is included as output_text."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "Let me reason step by step."}],
            }
        ]
        result = _translate_messages(messages)
        assert result[0]["content"] == [{"type": "output_text", "text": "Let me reason step by step."}]

    def test_native_thinking_signature_keeps_output_text_fallback(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Native provider reasoning.",
                        "signature": "native-anthropic-signature",
                    }
                ],
            }
        ]

        result = _translate_messages(messages)

        assert result == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Native provider reasoning."}],
            }
        ]

    def test_assistant_empty_thinking_block_skipped(self):
        """Assistant thinking block with empty thinking text is skipped."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": ""}],
            }
        ]
        result = _translate_messages(messages)
        assert result == []

    def test_encoded_reasoning_preserves_surrounding_text_order(self):
        signature = encode_reasoning_signature("rs_1", "encrypted-state")
        result = _translate_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Before"},
                        {"type": "thinking", "thinking": "Reason", "signature": signature},
                        {"type": "text", "text": "After"},
                    ],
                }
            ]
        )

        assert [item["type"] for item in result] == ["message", "reasoning", "message"]
        assert result[0]["content"][0]["text"] == "Before"
        assert result[2]["content"][0]["text"] == "After"

    def test_mixed_messages_ordering(self):
        """Full multi-turn conversation is converted in order."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_02",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_02",
                        "content": "Sunny, 72F",
                    }
                ],
            },
            {"role": "assistant", "content": "It's sunny and 72°F in NYC."},
        ]
        result = _translate_messages(messages)
        types = [item["type"] for item in result]
        assert types == ["message", "function_call", "function_call_output", "message"]

    def test_user_text_and_image_mixed(self):
        """User message with both text and image produces both parts."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.com/cat.jpg"},
                    },
                ],
            }
        ]
        result = _translate_messages(messages)
        assert len(result) == 1
        assert result[0]["content"][0] == {
            "type": "input_text",
            "text": "Describe this image:",
        }
        assert result[0]["content"][1] == {
            "type": "input_image",
            "image_url": "https://example.com/cat.jpg",
        }

    def test_malformed_litellm_signature_keeps_output_text_fallback(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Untrusted reasoning state.",
                        "signature": "litellm_openai_reasoning_v1_not-valid!",
                    }
                ],
            }
        ]

        result = _translate_messages(messages)

        assert result[0]["content"] == [{"type": "output_text", "text": "Untrusted reasoning state."}]

    def test_unknown_image_source_type_skipped(self):
        """Image block with unknown source type is silently skipped."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "file_path", "path": "/tmp/img.png"},
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result == []


# ---------------------------------------------------------------------------
# translate_tools_to_responses_api
# ---------------------------------------------------------------------------


class TestTranslateToolsToResponsesAPI:
    """Anthropic tool definitions -> Responses API function tools."""

    def test_regular_tool_with_description_and_schema(self):
        """Standard tool with description and input_schema is converted to function."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert result == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

    def test_tool_without_description(self):
        """Tool without a description omits the description key."""
        tools = [{"name": "ping", "input_schema": {"type": "object", "properties": {}}}]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "ping"
        assert "description" not in result[0]

    def test_tool_without_input_schema(self):
        """Tool without input_schema omits the parameters key."""
        tools = [{"name": "no_schema_tool", "description": "Does something."}]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert result[0]["type"] == "function"
        assert "parameters" not in result[0]

    def test_web_search_tool_by_name(self):
        """Tool named 'web_search' maps to web_search_preview."""
        tools = [{"name": "web_search", "type": "custom"}]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert result == [{"type": "web_search_preview"}]

    def test_web_search_tool_by_type_prefix(self):
        """Tool with type starting with 'web_search' maps to web_search_preview."""
        tools = [{"name": "search", "type": "web_search_20250305"}]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert result == [{"type": "web_search_preview"}]

    def test_multiple_tools_order_preserved(self):
        """Multiple tools are converted in order."""
        tools = [
            {"name": "tool_a", "description": "A"},
            {"name": "web_search", "type": "custom"},
            {"name": "tool_b", "description": "B"},
        ]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert len(result) == 3
        assert result[0]["name"] == "tool_a"
        assert result[1] == {"type": "web_search_preview"}
        assert result[2]["name"] == "tool_b"

    def test_empty_tools_list(self):
        """Empty tools list returns empty list."""
        assert _ADAPTER.translate_tools_to_responses_api([]) == []


# ---------------------------------------------------------------------------
# translate_tool_choice_to_responses_api
# ---------------------------------------------------------------------------


class TestTranslateToolChoiceToResponsesAPI:
    """Anthropic tool_choice -> Responses API tool_choice."""

    def test_auto_maps_to_auto(self):
        assert _ADAPTER.translate_tool_choice_to_responses_api({"type": "auto"}) == "auto"

    def test_any_maps_to_required(self):
        assert _ADAPTER.translate_tool_choice_to_responses_api({"type": "any"}) == "required"

    def test_specific_tool_maps_to_function(self):
        result = _ADAPTER.translate_tool_choice_to_responses_api({"type": "tool", "name": "get_weather"})
        assert result == {"type": "function", "name": "get_weather"}

    def test_none_maps_to_none(self):
        result = _ADAPTER.translate_tool_choice_to_responses_api({"type": "none"})
        assert result == "none"


# ---------------------------------------------------------------------------
# translate_thinking_to_reasoning
# ---------------------------------------------------------------------------


class TestTranslateThinkingToReasoning:
    """Anthropic thinking param -> Responses API reasoning param."""

    def test_budget_high_effort(self):
        result = _ADAPTER.translate_thinking_to_reasoning({"type": "enabled", "budget_tokens": 10000})
        # Default (reasoning_auto_summary=False): only effort, no summary
        assert result == {"effort": "high"}
        assert result is not None and "summary" not in result

    def test_budget_above_threshold_high_effort(self):
        result = _ADAPTER.translate_thinking_to_reasoning({"type": "enabled", "budget_tokens": 50000})
        assert result is not None
        assert result["effort"] == "high"
        assert "summary" not in result

    def test_budget_medium_effort(self):
        result = _ADAPTER.translate_thinking_to_reasoning(
            {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
            }
        )
        assert result == {"effort": "medium"}
        assert result is not None and "summary" not in result

    def test_budget_low_effort(self):
        result = _ADAPTER.translate_thinking_to_reasoning(
            {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
            }
        )
        assert result == {"effort": "low"}
        assert result is not None and "summary" not in result

    def test_budget_minimal_effort(self):
        result = _ADAPTER.translate_thinking_to_reasoning({"type": "enabled", "budget_tokens": 500})
        assert result == {"effort": "minimal"}
        assert result is not None and "summary" not in result

    def test_budget_at_exact_thresholds(self):
        result_high = _ADAPTER.translate_thinking_to_reasoning(
            {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
            }
        )
        assert result_high is not None
        assert result_high["effort"] == "high"
        result_medium = _ADAPTER.translate_thinking_to_reasoning(
            {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
            }
        )
        assert result_medium is not None
        assert result_medium["effort"] == "medium"
        assert "summary" not in result_medium
        result_low = _ADAPTER.translate_thinking_to_reasoning(
            {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
            }
        )
        assert result_low is not None
        assert result_low["effort"] == "low"
        assert "summary" not in result_low

    def test_disabled_type_returns_none(self):
        result = _ADAPTER.translate_thinking_to_reasoning({"type": "disabled"})
        assert result is None

    def test_non_dict_returns_none(self):
        result = _ADAPTER.translate_thinking_to_reasoning("enabled")  # type: ignore
        assert result is None

    def test_missing_budget_defaults_to_minimal(self):
        """Missing budget_tokens defaults to 0, below the low threshold -> minimal."""
        result = _ADAPTER.translate_thinking_to_reasoning({"type": "enabled"})
        assert result == {"effort": "minimal"}
        assert result is not None and "summary" not in result

    def test_summary_added_when_auto_summary_enabled(self):
        """When reasoning_auto_summary is True, summary='detailed' is included."""
        import litellm

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = True
            result = _ADAPTER.translate_thinking_to_reasoning({"type": "enabled", "budget_tokens": 10000})
            assert result == {"effort": "high", "summary": "detailed"}
        finally:
            litellm.reasoning_auto_summary = original

    def test_summary_added_when_env_var_set(self):
        """When LITELLM_REASONING_AUTO_SUMMARY env var is true, summary is included."""
        import litellm

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = False
            os.environ["LITELLM_REASONING_AUTO_SUMMARY"] = "true"
            result = _ADAPTER.translate_thinking_to_reasoning(
                {
                    "type": "enabled",
                    "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
                }
            )
            assert result == {"effort": "medium", "summary": "detailed"}
        finally:
            litellm.reasoning_auto_summary = original
            os.environ.pop("LITELLM_REASONING_AUTO_SUMMARY", None)


# ---------------------------------------------------------------------------
# translate_request – broader coverage
# ---------------------------------------------------------------------------


class TestTranslateRequestBroaderCoverage:
    """Full translate_request call: field-by-field mapping verification."""

    def test_model_and_input_always_present(self):
        req = _make_request()
        kwargs = _ADAPTER.translate_request(req)
        assert "model" in kwargs
        assert "input" in kwargs

    def test_system_string_becomes_instructions(self):
        req = _make_request(system="You are a helpful assistant.")
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["instructions"] == "You are a helpful assistant."

    def test_system_list_of_text_blocks_joined(self):
        req = _make_request(
            system=[
                {"type": "text", "text": "Be concise."},
                {"type": "text", "text": "Be helpful."},
            ]
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["instructions"] == "Be concise.\nBe helpful."

    def test_system_list_skips_non_text_blocks(self):
        req = _make_request(
            system=[
                {"type": "image", "source": {}},
                {"type": "text", "text": "Only text matters."},
            ]
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["instructions"] == "Only text matters."

    def test_system_list_strips_anthropic_billing_header(self):
        req = _make_request(
            system=[
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.207;",
                },
                {"type": "text", "text": "Real instructions"},
            ]
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["instructions"] == "Real instructions"

    def test_max_tokens_mapped_to_max_output_tokens(self):
        req = _make_request(max_tokens=512)
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["max_output_tokens"] == 512

    def test_max_tokens_clamped_to_openai_minimum(self):
        req = _make_request(max_tokens=1)
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["max_output_tokens"] == 16

    def test_temperature_passed_through(self):
        req = _make_request(temperature=0.7)
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["temperature"] == 0.7

    def test_top_p_passed_through(self):
        req = _make_request(top_p=0.9)
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["top_p"] == 0.9

    def test_tools_translated(self):
        req = _make_request(
            tools=[
                {
                    "name": "calculator",
                    "description": "Does math.",
                    "input_schema": {},
                    "strict": True,
                }
            ]
        )
        kwargs = _ADAPTER.translate_request(req)
        assert len(kwargs["tools"]) == 1
        assert kwargs["tools"][0]["name"] == "calculator"
        assert kwargs["tools"][0]["strict"] is True

    def test_tool_choice_translated(self):
        req = _make_request(
            tools=[{"name": "do_thing"}],
            tool_choice={"type": "tool", "name": "do_thing"},
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["tool_choice"] == {"type": "function", "name": "do_thing"}

    def test_long_tool_names_use_shared_hash_mapping(self):
        long_name = "tool_" + "x" * 80
        truncated_name = truncate_tool_name(long_name)
        req = _make_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_long",
                            "name": long_name,
                            "input": {},
                        }
                    ],
                }
            ],
            tools=[{"name": long_name}],
            tool_choice={"type": "tool", "name": long_name},
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["input"][0]["name"] == truncated_name
        assert kwargs["tools"][0]["name"] == truncated_name
        assert kwargs["tool_choice"]["name"] == truncated_name

    def test_disable_parallel_tool_use_translated(self):
        req = _make_request(
            tools=[{"name": "do_thing"}],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["parallel_tool_calls"] is False

    def test_thinking_translated_to_reasoning(self):
        req = _make_request(thinking={"type": "enabled", "budget_tokens": 12000})
        kwargs = _ADAPTER.translate_request(req)
        # reasoning_auto_summary is False by default, so no summary key
        assert kwargs["reasoning"] == {"effort": "high"}
        assert "summary" not in kwargs["reasoning"]

    def test_disabled_thinking_not_included_in_kwargs(self):
        req = _make_request(thinking={"type": "disabled"})
        kwargs = _ADAPTER.translate_request(req)
        assert "reasoning" not in kwargs

    def test_metadata_user_id_mapped_to_user(self):
        req = _make_request(metadata={"user_id": "user-42"})
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["user"] == "user-42"
        assert kwargs["prompt_cache_key"] == hashlib.sha256(b"user-42").hexdigest()

    def test_metadata_user_id_truncated_to_64_chars(self):
        first_id = "x" * 100
        second_id = "x" * 99 + "y"
        first_kwargs = _ADAPTER.translate_request(_make_request(metadata={"user_id": first_id}))
        second_kwargs = _ADAPTER.translate_request(_make_request(metadata={"user_id": second_id}))
        assert first_kwargs["user"] == second_kwargs["user"] == "x" * 64
        assert first_kwargs["prompt_cache_key"] == hashlib.sha256(first_id.encode()).hexdigest()
        assert first_kwargs["prompt_cache_key"] != second_kwargs["prompt_cache_key"]

    def test_no_optional_fields_does_not_add_spurious_keys(self):
        req = _make_request()
        kwargs = _ADAPTER.translate_request(req)
        for key in (
            "instructions",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "reasoning",
            "text",
            "context_management",
            "user",
            "prompt_cache_key",
            "parallel_tool_calls",
        ):
            assert key not in kwargs, f"unexpected key: {key}"

    def test_requests_encrypted_reasoning_content(self):
        kwargs = _ADAPTER.translate_request(_make_request())

        assert kwargs["include"] == ["reasoning.encrypted_content"]


# ---------------------------------------------------------------------------
# translate_response
# ---------------------------------------------------------------------------


def _make_mock_response(
    output: list,
    status: str = "completed",
    response_id: str = "resp_001",
    model: str = "gpt-4o",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    """Build a minimal mock ResponsesAPIResponse."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    resp = MagicMock()
    resp.id = response_id
    resp.model = model
    resp.status = status
    resp.output = output
    resp.usage = usage
    resp.usage.input_tokens_details = None
    return resp


def _make_output_message(texts: List[str]) -> MagicMock:
    """Build a mock ResponseOutputMessage with output_text parts."""
    from openai.types.responses import ResponseOutputMessage  # type: ignore[import]

    parts = []
    for t in texts:
        part = MagicMock()
        part.type = "output_text"
        part.text = t
        parts.append(part)

    msg = MagicMock(spec=ResponseOutputMessage)
    msg.content = parts
    return msg


def _make_function_call_item(call_id: str, name: str, arguments: str) -> MagicMock:
    """Build a mock ResponseFunctionToolCall."""
    from openai.types.responses import ResponseFunctionToolCall  # type: ignore[import]

    item = MagicMock(spec=ResponseFunctionToolCall)
    item.call_id = call_id
    item.id = call_id
    item.name = name
    item.arguments = arguments
    return item


def _make_reasoning_item(
    summaries: List[str],
    item_id: str = "",
    encrypted_content: str | None = None,
) -> MagicMock:
    """Build a mock ResponseReasoningItem."""
    from openai.types.responses import ResponseReasoningItem  # type: ignore[import]

    summary_mocks = []
    for text in summaries:
        s = MagicMock()
        s.text = text
        summary_mocks.append(s)

    item = MagicMock(spec=ResponseReasoningItem)
    item.type = "reasoning"
    item.id = item_id
    item.encrypted_content = encrypted_content
    item.summary = summary_mocks
    return item


class TestTranslateResponse:
    """Responses API -> AnthropicMessagesResponse conversion."""

    def test_output_text_message_becomes_text_block(self):
        """ResponseOutputMessage with output_text parts -> Anthropic text content."""
        response = _make_mock_response(output=[_make_output_message(["Hello!"])])
        result: Any = _ADAPTER.translate_response(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"

    def test_multiple_text_parts(self):
        """Multiple output_text parts become multiple text content blocks."""
        response = _make_mock_response(output=[_make_output_message(["Part 1", "Part 2"])])
        result: Any = _ADAPTER.translate_response(response)
        assert len(result["content"]) == 2
        assert result["content"][0]["text"] == "Part 1"
        assert result["content"][1]["text"] == "Part 2"

    def test_function_call_becomes_tool_use(self):
        """ResponseFunctionToolCall -> Anthropic tool_use content block."""
        fc = _make_function_call_item("call_99", "get_weather", '{"city": "NYC"}')
        response = _make_mock_response(output=[fc])
        result: Any = _ADAPTER.translate_response(response)
        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call_99"
        assert block["name"] == "get_weather"
        assert block["input"] == {"city": "NYC"}

    def test_function_call_sets_stop_reason_tool_use(self):
        """Presence of a function_call sets stop_reason to 'tool_use'."""
        fc = _make_function_call_item("call_1", "tool_a", "{}")
        response = _make_mock_response(output=[fc])
        result: Any = _ADAPTER.translate_response(response)
        assert result["stop_reason"] == "tool_use"

    def test_function_call_restores_long_tool_name(self):
        original_name = "tool_" + "x" * 80
        truncated_name = truncate_tool_name(original_name)
        fc = _make_function_call_item("call_1", truncated_name, "{}")
        response = _make_mock_response(output=[fc])
        result: Any = _ADAPTER.translate_response(
            response,
            tool_name_mapping={truncated_name: original_name},
        )
        assert result["content"][0]["name"] == original_name

    def test_text_only_stop_reason_end_turn(self):
        """Text-only response has stop_reason 'end_turn'."""
        response = _make_mock_response(output=[_make_output_message(["Hi"])])
        result: Any = _ADAPTER.translate_response(response)
        assert result["stop_reason"] == "end_turn"

    def test_refusal_is_preserved_as_text_and_stop_reason(self):
        response = _make_mock_response(
            output=[
                {
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
                }
            ]
        )

        result: Any = _ADAPTER.translate_response(response)

        assert result["content"] == [{"type": "text", "text": "I cannot help with that."}]
        assert result["stop_reason"] == "refusal"

    def test_incomplete_status_sets_max_tokens(self):
        """status='incomplete' overrides stop_reason to 'max_tokens'."""
        response = _make_mock_response(
            output=[_make_output_message(["Truncated..."])],
            status="incomplete",
        )
        result: Any = _ADAPTER.translate_response(response)
        assert result["stop_reason"] == "max_tokens"

    def test_incomplete_content_filter_preserves_refusal_stop_reason(self):
        response = _make_mock_response(
            output=[
                {
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
                }
            ],
            status="incomplete",
        )
        response.incomplete_details = SimpleNamespace(reason="content_filter")

        result: Any = _ADAPTER.translate_response(response)

        assert result["content"] == [{"type": "text", "text": "I cannot help with that."}]
        assert result["stop_reason"] == "refusal"

    def test_reasoning_item_becomes_thinking_block(self):
        """ResponseReasoningItem summaries -> Anthropic thinking content blocks."""
        reasoning = _make_reasoning_item(["Step 1: analyze. Step 2: conclude."])
        response = _make_mock_response(output=[reasoning])
        result: Any = _ADAPTER.translate_response(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "thinking"
        assert "Step 1" in result["content"][0]["thinking"]

    def test_reasoning_item_signature_round_trips_encrypted_state(self):
        reasoning = _make_reasoning_item(
            ["Inspect the tool result."],
            item_id="rs_123",
            encrypted_content="encrypted-reasoning-state",
        )
        response = _make_mock_response(output=[reasoning])

        result: Any = _ADAPTER.translate_response(response)
        thinking_block = result["content"][0]
        replayed = _translate_messages([{"role": "assistant", "content": [thinking_block]}])

        assert isinstance(thinking_block["signature"], str)
        assert replayed == [
            {
                "type": "reasoning",
                "id": "rs_123",
                "encrypted_content": "encrypted-reasoning-state",
                "summary": [{"type": "summary_text", "text": "Inspect the tool result."}],
            }
        ]

    def test_empty_reasoning_summary_skipped(self):
        """Reasoning item with empty text summary is not added to content."""
        reasoning = _make_reasoning_item([""])
        response = _make_mock_response(output=[reasoning])
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"] == []

    def test_usage_mapped_correctly(self):
        """Input/output tokens from ResponseAPIUsage are mapped to AnthropicUsage."""
        response = _make_mock_response(
            output=[_make_output_message(["OK"])],
            input_tokens=200,
            output_tokens=75,
        )
        result: Any = _ADAPTER.translate_response(response)
        assert result["usage"]["input_tokens"] == 200
        assert result["usage"]["output_tokens"] == 75

    def test_usage_splits_cache_tokens_from_object_details(self):
        response = _make_mock_response(output=[], input_tokens=100, output_tokens=25)
        response.usage.input_tokens_details = SimpleNamespace(
            cached_tokens=40,
            cache_write_tokens=5,
        )
        result: Any = _ADAPTER.translate_response(response)
        assert result["usage"] == {
            "input_tokens": 55,
            "output_tokens": 25,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 40,
        }

    def test_usage_splits_cache_tokens_from_dict_details(self):
        response = _make_mock_response(output=[], input_tokens=100, output_tokens=25)
        response.usage.input_tokens_details = {
            "cached_tokens": 40,
            "cache_creation_tokens": 5,
        }
        result: Any = _ADAPTER.translate_response(response)
        assert result["usage"] == {
            "input_tokens": 55,
            "output_tokens": 25,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 40,
        }

    def test_model_and_id_preserved(self):
        """Model and response ID from the Responses API are forwarded."""
        response = _make_mock_response(
            output=[_make_output_message(["Hi"])],
            response_id="resp_xyz",
            model="gpt-4-turbo",
        )
        result: Any = _ADAPTER.translate_response(response)
        assert result["id"] == "resp_xyz"
        assert result["model"] == "gpt-4-turbo"

    def test_role_is_always_assistant(self):
        response = _make_mock_response(output=[_make_output_message(["Hi"])])
        result: Any = _ADAPTER.translate_response(response)
        assert result["role"] == "assistant"

    def test_type_is_always_message(self):
        response = _make_mock_response(output=[_make_output_message(["Hi"])])
        result: Any = _ADAPTER.translate_response(response)
        assert result["type"] == "message"

    def test_empty_output_list(self):
        """Empty output list produces empty content with 'end_turn' stop reason."""
        response = _make_mock_response(output=[])
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"] == []
        assert result["stop_reason"] == "end_turn"

    def test_function_call_with_invalid_json_arguments(self):
        """Invalid JSON in function_call arguments falls back to empty dict."""
        fc = _make_function_call_item("call_bad", "broken_tool", "not-valid-json")
        response = _make_mock_response(output=[fc])
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"][0]["input"] == {}

    def test_dict_output_message_item(self):
        """Dict-shaped output message (type=message) is also handled."""
        output_item = {
            "type": "message",
            "content": [{"type": "output_text", "text": "Dict-based response"}],
        }
        response = _make_mock_response(output=[output_item])
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Dict-based response"

    def test_dict_function_call_item(self):
        """Dict-shaped function_call item is converted to tool_use block."""
        output_item = {
            "type": "function_call",
            "call_id": "call_dict_1",
            "name": "search",
            "arguments": '{"query": "cats"}',
        }
        response = _make_mock_response(output=[output_item])
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "search"
        assert result["content"][0]["input"] == {"query": "cats"}
        assert result["stop_reason"] == "tool_use"

    def test_mixed_reasoning_text_and_tool_use(self):
        """Reasoning + text + tool_use in one response all convert correctly."""
        reasoning = _make_reasoning_item(["Thinking..."])
        text_msg = _make_output_message(["Here is my answer."])
        fc = _make_function_call_item("call_mix", "lookup", '{"id": 1}')
        response = _make_mock_response(output=[reasoning, text_msg, fc])
        result: Any = _ADAPTER.translate_response(response)
        types = [b["type"] for b in result["content"]]
        assert "thinking" in types
        assert "text" in types
        assert "tool_use" in types
        assert result["stop_reason"] == "tool_use"

    def test_stop_sequence_truncates_text_and_sets_stop_reason(self):
        """A stop sequence in output text truncates the text and sets stop_reason."""
        response = _make_mock_response(output=[_make_output_message(["keep this </block> drop this"])])
        result: Any = _ADAPTER.translate_response(response, stop_sequences=["</block>"])
        assert result["content"][0]["text"] == "keep this "
        assert result["stop_reason"] == "stop_sequence"
        assert result["stop_sequence"] == "</block>"

    def test_stop_sequence_drops_later_content_blocks(self):
        """Everything after the stop sequence, including later blocks, is removed."""
        text_msg = _make_output_message(["answer STOP"])
        fc = _make_function_call_item("call_1", "tool_a", "{}")
        response = _make_mock_response(output=[text_msg, fc])
        result: Any = _ADAPTER.translate_response(response, stop_sequences=["STOP"])
        assert [b["type"] for b in result["content"]] == ["text"]
        assert result["content"][0]["text"] == "answer "
        assert result["stop_reason"] == "stop_sequence"
        assert result["stop_sequence"] == "STOP"

    def test_stop_sequence_absent_leaves_response_unchanged(self):
        response = _make_mock_response(output=[_make_output_message(["no marker here"])])
        result: Any = _ADAPTER.translate_response(response, stop_sequences=["</block>"])
        assert result["content"][0]["text"] == "no marker here"
        assert result["stop_reason"] == "end_turn"
        assert result["stop_sequence"] is None

    def test_stop_sequence_not_applied_on_refusal(self):
        """Refusal takes precedence over an emulated stop sequence."""
        response = _make_mock_response(
            output=[
                {
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": "I cannot STOP help."}],
                }
            ],
        )
        result: Any = _ADAPTER.translate_response(response, stop_sequences=["STOP"])
        assert result["stop_reason"] == "refusal"
        assert result["content"][0]["text"] == "I cannot STOP help."

    def test_service_tier_mapped_into_usage(self):
        response = _make_mock_response(output=[_make_output_message(["ok"])])
        response.service_tier = "default"
        result: Any = _ADAPTER.translate_response(response)
        assert result["usage"]["service_tier"] == "standard"

    def test_model_context_window_exceeded_maps_stop_reason(self):
        response = _make_mock_response(
            output=[_make_output_message(["partial"])],
            status="incomplete",
        )
        response.incomplete_details = SimpleNamespace(reason="model_context_window_exceeded")
        result: Any = _ADAPTER.translate_response(response)
        assert result["stop_reason"] == "model_context_window_exceeded"
