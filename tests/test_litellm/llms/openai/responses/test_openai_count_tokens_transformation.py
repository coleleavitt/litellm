import os
import sys

from openai.types.responses.input_token_count_params import InputTokenCountParams
from pydantic import TypeAdapter

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path
from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
)


def test_transform_basic_request():
    """Test basic request with model and input."""
    config = OpenAICountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="gpt-4o",
        input="Hello, how are you?",
    )

    assert result == {
        "model": "gpt-4o",
        "input": "Hello, how are you?",
    }


def test_transform_with_list_input():
    """Test request with list input format."""
    config = OpenAICountTokensConfig()

    input_items = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = config.transform_request_to_count_tokens(
        model="gpt-4o",
        input=input_items,
    )

    assert result["model"] == "gpt-4o"
    assert result["input"] == input_items


def test_transform_includes_instructions():
    """Test that instructions are included when provided."""
    config = OpenAICountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="gpt-4o",
        input="Hello",
        instructions="You are a helpful assistant.",
    )

    assert result["instructions"] == "You are a helpful assistant."
    assert result["model"] == "gpt-4o"
    assert result["input"] == "Hello"


def test_transform_includes_tools():
    """Test that tools are included when provided."""
    config = OpenAICountTokensConfig()

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]

    result = config.transform_request_to_count_tokens(
        model="gpt-4o",
        input="What's the weather?",
        tools=tools,
    )

    assert result["tools"] == tools


def test_transform_no_instructions_no_tools():
    """Test that None values are not included."""
    config = OpenAICountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="gpt-4o",
        input="Hello",
        instructions=None,
        tools=None,
    )

    assert "instructions" not in result
    assert "tools" not in result


def test_messages_to_responses_input_basic():
    """Test converting basic chat messages to Responses API input format."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(messages)

    assert len(input_items) == 3
    assert input_items[0] == {"role": "user", "content": "Hello"}
    assert input_items[1] == {"role": "assistant", "content": "Hi there!"}
    assert input_items[2] == {"role": "user", "content": "How are you?"}
    assert instructions is None


def test_messages_to_responses_input_with_system():
    """Test that system messages are extracted as instructions."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(messages)

    assert len(input_items) == 1
    assert input_items[0] == {"role": "user", "content": "Hello"}
    assert instructions == "You are helpful."


def test_messages_to_responses_input_with_developer():
    """Test that developer messages are extracted as instructions."""
    messages = [
        {"role": "developer", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]

    input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(messages)

    assert len(input_items) == 1
    assert instructions == "Be concise."


def test_messages_to_responses_input_with_tool():
    """Test that tool messages are converted to function_call_output."""
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {"role": "tool", "content": "72°F", "tool_call_id": "call_123"},
    ]

    input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(messages)

    assert len(input_items) == 2
    assert input_items[1] == {
        "type": "function_call_output",
        "call_id": "call_123",
        "output": "72°F",
    }


def test_anthropic_history_maps_to_responses_count_request_without_reordering():
    config = OpenAICountTokensConfig()
    system = [
        {
            "type": "text",
            "text": "Count accurately.",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "Keep tool history."},
    ]
    tools = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect this image."},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "aGVsbG8=",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I should read the file.",
                    "signature": "signature",
                },
                {"type": "text", "text": "Reading it now."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "read_file",
                    "input": {"path": "/tmp/example"},
                },
                {"type": "text", "text": "The call was queued."},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": [{"type": "text", "text": "file contents"}],
                },
                {"type": "text", "text": "Continue."},
            ],
        },
    ]

    input_items, instructions = config.messages_to_responses_input(
        messages,
        tools=tools,
        system=system,
    )
    result = config.transform_request_to_count_tokens(
        model="gpt-5.6-sol",
        input=input_items,
        tools=tools,
        instructions=instructions,
    )

    assert result == {
        "model": "gpt-5.6-sol",
        "instructions": "Count accurately.\nKeep tool history.",
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Inspect this image."},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,aGVsbG8=",
                        "detail": "auto",
                    },
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "input_text", "text": "I should read the file."},
                    {"type": "input_text", "text": "Reading it now."},
                ],
            },
            {
                "type": "function_call",
                "call_id": "toolu_123",
                "name": "read_file",
                "arguments": '{"path": "/tmp/example"}',
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "The call was queued."}],
            },
            {
                "type": "function_call_output",
                "call_id": "toolu_123",
                "output": [{"type": "input_text", "text": "file contents"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue."}],
            },
        ],
    }
    TypeAdapter(InputTokenCountParams).validate_python(result)


def test_anthropic_count_instructions_strip_billing_header():
    _, instructions = OpenAICountTokensConfig.messages_to_responses_input(
        [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        system=[
            {"type": "text", "text": "x-anthropic-billing-header: synthetic"},
            {"type": "text", "text": "Real instructions"},
        ],
    )

    assert instructions == "Real instructions"


def test_validate_request_valid():
    """Test that valid requests pass validation."""
    config = OpenAICountTokensConfig()
    config.validate_request(model="gpt-4o", input="Hello")


def test_validate_request_missing_model():
    """Test that missing model raises ValueError."""
    config = OpenAICountTokensConfig()
    try:
        config.validate_request(model="", input="Hello")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "model" in str(e)


def test_validate_request_missing_input():
    """Test that missing input raises ValueError."""
    config = OpenAICountTokensConfig()
    try:
        config.validate_request(model="gpt-4o", input="")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "input" in str(e)


def test_get_endpoint_default():
    """Test default endpoint URL."""
    config = OpenAICountTokensConfig()
    assert config.get_openai_count_tokens_endpoint() == "https://api.openai.com/v1/responses/input_tokens"


def test_get_endpoint_custom_base():
    """Test custom API base URL."""
    config = OpenAICountTokensConfig()
    assert (
        config.get_openai_count_tokens_endpoint("https://custom.api.com/v1")
        == "https://custom.api.com/v1/responses/input_tokens"
    )


def test_get_required_headers():
    """Test required headers include Authorization."""
    config = OpenAICountTokensConfig()
    headers = config.get_required_headers("sk-test-key")

    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["Content-Type"] == "application/json"
