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
