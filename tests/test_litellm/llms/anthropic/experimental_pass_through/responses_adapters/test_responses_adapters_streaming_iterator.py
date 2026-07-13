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


def test_seeded_prompt_tokens_single_message_start_with_tool_use():
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=None,
        model="m",
        prompt_tokens=57,
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
    # Exactly one message_start. A second one mid-stream violates the Anthropic
    # SSE contract and trips the @anthropic-ai/sdk stream accumulator.
    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    # message_start carries the up-front prompt-token estimate so the live
    # counter shows a real number immediately instead of jumping from 0.
    assert chunks[0]["message"]["usage"] == {
        "input_tokens": 57,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    # message_delta carries the authoritative usage from response.completed.
    assert chunks[4]["usage"] == {
        "input_tokens": 12,
        "output_tokens": 21,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 40,
    }
    assert chunks[4]["delta"]["stop_reason"] == "tool_use"


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


def test_single_message_start_across_multiple_blocks():
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=None,
        model="m",
        prompt_tokens=57,
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
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert [chunk["type"] for chunk in chunks].count("message_start") == 1
    assert chunks[0]["message"]["usage"]["input_tokens"] == 57
    assert chunks[5]["usage"] == {
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
            "type": "content_block_stop",
            "index": 0,
        },
        {
            "type": "error",
            "error": {"type": "api_error", "message": "upstream failed"},
            "request_id": "response_1",
        },
    ]


async def test_upstream_eof_emits_error_after_completed_blocks():
    async def events():
        yield {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_1", "call_id": "call_1", "name": "ping"},
        }
        yield {"type": "response.output_item.done", "item": {"type": "function_call", "id": "item_1"}}

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=events(),
        model="m",
    )
    chunks = [chunk async for chunk in wrapper]

    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "error",
    ]
    assert chunks[-1]["error"]["message"] == "Upstream response ended before a terminal event"


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


def test_refusal_streams_as_text_with_refusal_stop_reason():
    chunks = _process_all(
        [
            {"type": "response.output_item.added", "item": {"type": "message", "id": "message_1"}},
            {"type": "response.refusal.delta", "item_id": "message_1", "delta": "I cannot help with that."},
            {"type": "response.output_item.done", "item": {"type": "message", "id": "message_1"}},
            {
                "type": "response.completed",
                "response": SimpleNamespace(
                    status="completed",
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(type="refusal", refusal="I cannot help with that.")],
                        )
                    ],
                    usage=None,
                ),
            },
        ]
    )

    assert chunks[1]["delta"] == {"type": "text_delta", "text": "I cannot help with that."}
    assert chunks[-2]["delta"]["stop_reason"] == "refusal"


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


def _completed_event(**response_kwargs):
    response_kwargs.setdefault("status", "completed")
    response_kwargs.setdefault(
        "usage",
        SimpleNamespace(input_tokens=1, output_tokens=1, input_tokens_details=None),
    )
    response_kwargs.setdefault("output", [])
    return {"type": "response.completed", "response": SimpleNamespace(**response_kwargs)}


def _run_with_stop(stop_sequences, deltas):
    """Stream a single text block made of ``deltas`` under ``stop_sequences``."""
    events = [
        {"type": "response.created"},
        {"type": "response.output_item.added", "item": {"type": "message", "id": "m1"}},
        *[{"type": "response.output_text.delta", "item_id": "m1", "delta": d} for d in deltas],
        {"type": "response.output_item.done", "item": {"type": "message", "id": "m1"}},
        _completed_event(),
    ]
    wrapper = AnthropicResponsesStreamWrapper(responses_stream=None, model="m", stop_sequences=stop_sequences)
    for event in events:
        wrapper._process_event(event)
    chunks = list(wrapper._chunk_queue)
    text = "".join(c["delta"]["text"] for c in chunks if c["type"] == "content_block_delta")
    message_delta = next(c for c in chunks if c["type"] == "message_delta")
    return text, message_delta, chunks


def test_stop_sequence_truncates_streamed_text():
    text, message_delta, _ = _run_with_stop(["STOP"], ["hello ", "world STOP tail"])
    assert text == "hello world "
    assert "tail" not in text
    assert message_delta["delta"]["stop_reason"] == "stop_sequence"
    assert message_delta["delta"]["stop_sequence"] == "STOP"


def test_stop_sequence_spanning_two_deltas():
    text, message_delta, _ = _run_with_stop(["STOP"], ["abc ST", "OP xyz"])
    assert text == "abc "
    assert message_delta["delta"]["stop_reason"] == "stop_sequence"


def test_stop_sequence_partial_suffix_flushed_when_no_match():
    """A trailing partial match that never completes is flushed at block done."""
    text, message_delta, _ = _run_with_stop(["STOP"], ["abcST"])
    assert text == "abcST"
    assert message_delta["delta"]["stop_reason"] == "end_turn"
    assert message_delta["delta"]["stop_sequence"] is None


def test_stop_sequence_closes_block_and_suppresses_later_items():
    """After a stop, a later output item (e.g. tool call) is not emitted."""
    events = [
        {"type": "response.created"},
        {"type": "response.output_item.added", "item": {"type": "message", "id": "m1"}},
        {"type": "response.output_text.delta", "item_id": "m1", "delta": "done STOP"},
        {"type": "response.output_item.done", "item": {"type": "message", "id": "m1"}},
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "f1", "call_id": "c1", "name": "ping"},
        },
        {"type": "response.output_item.done", "item": {"type": "function_call", "id": "f1"}},
        _completed_event(output=[SimpleNamespace(type="function_call")]),
    ]
    wrapper = AnthropicResponsesStreamWrapper(responses_stream=None, model="m", stop_sequences=["STOP"])
    for event in events:
        wrapper._process_event(event)
    chunks = list(wrapper._chunk_queue)
    types = [c["type"] for c in chunks]
    # only one content block (the text) — the tool_use block is suppressed
    assert types.count("content_block_start") == 1
    assert chunks[1]["content_block"]["type"] == "text"
    message_delta = next(c for c in chunks if c["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "stop_sequence"


def test_no_stop_sequences_emits_text_verbatim():
    """Without stop_sequences the emulation is a no-op even if the string appears."""
    text, message_delta, _ = _run_with_stop(None, ["STOP stays here"])
    assert text == "STOP stays here"
    assert message_delta["delta"]["stop_reason"] == "end_turn"


def test_service_tier_echoed_in_message_delta():
    events = [
        {"type": "response.created"},
        _completed_event(service_tier="priority"),
    ]
    chunks = _process_all(events)
    message_delta = next(c for c in chunks if c["type"] == "message_delta")
    assert message_delta["usage"]["service_tier"] == "priority"


def test_model_context_window_exceeded_streaming_stop_reason():
    events = [
        {"type": "response.created"},
        {
            "type": "response.incomplete",
            "response": SimpleNamespace(
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="model_context_window_exceeded"),
                usage=SimpleNamespace(input_tokens=9, output_tokens=0, input_tokens_details=None),
                output=[],
            ),
        },
    ]
    chunks = _process_all(events)
    message_delta = next(c for c in chunks if c["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "model_context_window_exceeded"


def test_mcp_call_streams_as_mcp_tool_blocks():
    events = [
        {"type": "response.created"},
        {"type": "response.output_item.added", "item": {"type": "mcp_call", "id": "c1"}},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "mcp_call",
                "id": "c1",
                "name": "get_forecast",
                "server_label": "weather",
                "arguments": '{"city": "NYC"}',
                "output": "sunny",
            },
        },
        _completed_event(),
    ]
    chunks = _process_all(events)
    starts = [c for c in chunks if c["type"] == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["mcp_tool_use", "mcp_tool_result"]
    assert starts[0]["content_block"]["input"] == {"city": "NYC"}
    assert starts[1]["content_block"]["content"] == [{"type": "text", "text": "sunny"}]
    # each block is opened and closed
    assert [c["type"] for c in chunks if c["type"] in ("content_block_start", "content_block_stop")] == [
        "content_block_start",
        "content_block_stop",
        "content_block_start",
        "content_block_stop",
    ]
    message_delta = next(c for c in chunks if c["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "end_turn"
