"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)


def _process_all(events: list) -> list:
    wrapper = AnthropicResponsesStreamWrapper(responses_stream=None, model="m")
    for event in events:
        wrapper._process_event(event)
    return list(wrapper._chunk_queue)


def test_claude_code_per_turn_usage_precedes_tool_block_stop():
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=None,
        model="m",
        claude_code_per_turn_usage=True,
    )
    events = [
        {"type": "response.created"},
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_1", "call_id": "call_1", "name": "ping"},
        },
        {"type": "response.function_call_arguments.delta", "item_id": "item_1", "delta": '{"value":"ok"}'},
        {"type": "response.output_item.done", "item": {"type": "function_call", "id": "item_1"}},
        {
            "type": "response.completed",
            "response": SimpleNamespace(
                status="completed",
                usage=SimpleNamespace(
                    input_tokens=57,
                    output_tokens=21,
                    input_tokens_details=SimpleNamespace(cached_tokens=40, cache_write_tokens=5),
                ),
                output=[SimpleNamespace(type="function_call")],
            ),
        },
    ]

    for event in events:
        wrapper._process_event(event)

    chunks = list(wrapper._chunk_queue)
    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "message_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert chunks[3]["message"]["usage"] == {
        "input_tokens": 12,
        "output_tokens": 21,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 40,
    }
    assert chunks[5]["usage"] == {
        "input_tokens": 12,
        "output_tokens": 21,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 40,
    }
    assert chunks[5]["delta"]["stop_reason"] == "tool_use"


def test_stream_restores_truncated_tool_name():
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=None,
        model="m",
        tool_name_mapping={"truncated_name": "original_long_tool_name"},
    )
    wrapper._process_event(
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "item_1",
                "call_id": "call_1",
                "name": "truncated_name",
            },
        }
    )

    chunk = wrapper._chunk_queue.popleft()
    assert chunk["content_block"]["name"] == "original_long_tool_name"


def test_claude_code_per_turn_usage_is_attached_only_to_final_block_stop():
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=None,
        model="m",
        claude_code_per_turn_usage=True,
    )
    events = [
        {"type": "response.created"},
        {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "reasoning_1"}},
        {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "reasoning_1"}},
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_1", "call_id": "call_1", "name": "ping"},
        },
        {"type": "response.output_item.done", "item": {"type": "function_call", "id": "item_1"}},
        {
            "type": "response.completed",
            "response": SimpleNamespace(
                status="completed",
                usage=SimpleNamespace(
                    input_tokens=57,
                    output_tokens=21,
                    input_tokens_details=SimpleNamespace(cached_tokens=40, cache_write_tokens=5),
                ),
                output=[SimpleNamespace(type="function_call")],
            ),
        },
    ]

    for event in events:
        wrapper._process_event(event)

    chunks = list(wrapper._chunk_queue)
    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "content_block_start",
        "message_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert [chunk["type"] for chunk in chunks].count("message_start") == 2
    assert chunks[4]["message"]["usage"] == {
        "input_tokens": 12,
        "output_tokens": 21,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 40,
    }


def test_unmapped_output_item_done_does_not_close_the_current_block():
    chunks = _process_all(
        [
            {"type": "response.output_item.added", "item": {"type": "message", "id": "message_1"}},
            {"type": "response.output_item.done", "item": {"type": "image_generation_call", "id": "image_1"}},
        ]
    )

    assert chunks == [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
    ]


def test_response_failed_emits_error_without_success_terminal_events():
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=None,
        model="m",
        claude_code_per_turn_usage=True,
    )
    events = [
        {"type": "response.output_item.added", "item": {"type": "message", "id": "message_1"}},
        {"type": "response.output_item.done", "item": {"type": "message", "id": "message_1"}},
        {
            "type": "response.failed",
            "response": SimpleNamespace(
                id="response_1",
                error=SimpleNamespace(code="server_error", message="upstream failed"),
            ),
        },
    ]

    for event in events:
        wrapper._process_event(event)

    chunks = list(wrapper._chunk_queue)
    assert chunks == [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "error",
            "error": {"type": "api_error", "message": "upstream failed"},
            "request_id": "response_1",
        },
    ]
    assert wrapper._held_content_block_stop is None


def test_top_level_error_emits_anthropic_error():
    events = (
        {"type": "error", "code": "server_error", "message": "stream failed"},
        {"type": "error", "error": {"code": "server_error", "message": "stream failed"}},
    )

    for event in events:
        assert _process_all([event]) == [
            {
                "type": "error",
                "error": {"type": "api_error", "message": "stream failed"},
            }
        ]


def test_incomplete_reason_maps_to_anthropic_stop_reason():
    for incomplete_reason, stop_reason in (
        ("max_output_tokens", "max_tokens"),
        ("content_filter", "refusal"),
        ("refusal", "refusal"),
        ("unknown", "max_tokens"),
    ):
        chunks = _process_all(
            [
                {
                    "type": "response.incomplete",
                    "response": SimpleNamespace(
                        status="incomplete",
                        incomplete_details=SimpleNamespace(reason=incomplete_reason),
                        output=[],
                        usage=None,
                    ),
                }
            ]
        )

        assert chunks[0]["delta"]["stop_reason"] == stop_reason
        assert chunks[1] == {"type": "message_stop"}


def test_dict_terminal_usage_supports_cache_creation_aliases():
    for cache_creation_key in ("cache_write_tokens", "cache_creation_tokens"):
        chunks = _process_all(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [],
                        "usage": {
                            "input_tokens": 50,
                            "output_tokens": 7,
                            "input_tokens_details": {
                                "cached_tokens": 30,
                                cache_creation_key: 5,
                            },
                        },
                    },
                }
            ]
        )

        assert chunks[0]["usage"] == {
            "input_tokens": 15,
            "output_tokens": 7,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 30,
        }


def test_reasoning_signature_streams_before_stop_and_round_trips():
    wrapper = AnthropicResponsesStreamWrapper(responses_stream=None, model="m")
    events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "reasoning", "id": "rs_stream"},
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_stream",
            "delta": "Check the repository state.",
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "rs_stream",
                "encrypted_content": "encrypted-stream-state",
            },
        },
    ]

    for event in events:
        wrapper._process_event(event)

    chunks = list(wrapper._chunk_queue)
    assert [chunk["type"] for chunk in chunks] == [
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
    ]
    assert chunks[2]["delta"]["type"] == "signature_delta"
    adapter = LiteLLMAnthropicToResponsesAPIAdapter()
    replayed = adapter.translate_messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Check the repository state.",
                        "signature": chunks[2]["delta"]["signature"],
                    }
                ],
            }
        ]
    )
    assert replayed == [
        {
            "type": "reasoning",
            "id": "rs_stream",
            "encrypted_content": "encrypted-stream-state",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "Check the repository state.",
                }
            ],
        }
    ]


class TestProcessEventTextDeltaWithoutOutputItemAdded:
    """Streams that skip response.output_item.added (e.g. LMStudio) must still
    open a text block before any delta and never emit index -1."""

    def test_process_event_synthesizes_content_block_start_before_delta(self):
        chunks = _process_all(
            [
                {"type": "response.output_text.delta", "item_id": "i1", "delta": "Hel"},
                {"type": "response.output_text.delta", "item_id": "i1", "delta": "lo"},
            ]
        )
        assert [c["type"] for c in chunks] == [
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
        ]
        assert chunks[0]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks] == [0, 0, 0]
        assert chunks[1]["delta"] == {"type": "text_delta", "text": "Hel"}

    def test_process_event_delta_without_item_id_never_yields_negative_index(self):
        chunks = _process_all([{"type": "response.output_text.delta", "delta": "Hi"}])
        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
        ]

    def test_process_event_unregistered_item_id_opens_new_text_block(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "reasoning", "id": "rs_1"},
                },
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )
        assert chunks[1]["type"] == "content_block_start"
        assert chunks[1]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks[1:]] == [1, 1]

    def test_process_event_registered_item_id_does_not_synthesize_start(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "message", "id": "m1"},
                },
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )
        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
        ]
