from raycast_gateway.adapters import (
    GatewayDefaults,
    ResponsesStreamState,
    StreamState,
    aggregate_openai_response,
    aggregate_responses_response,
    company_sse_data_to_dict,
    encode_sse_data,
    encode_sse_event,
    estimate_input_tokens,
    final_response_stream_events,
    internal_chunk_to_openai_chunks,
    internal_chunk_to_response_events,
    normalize_finish_reason,
    normalize_responses_usage,
    openai_request_to_company_payload,
    raycast_model_catalog,
    raycast_models_to_openai_models,
    response_created_event,
    responses_request_to_chat_payload,
    resolve_model_for_company,
    resolve_reasoning_effort_for_company,
    split_model_for_company,
)


def test_openai_request_to_company_payload_maps_messages_and_tools():
    payload = {
        "model": "google/gemini-3.5-flash",
        "messages": [{"role": "user", "content": "今天几号?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "location-get-current-location",
                    "description": "Gets current location.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    }

    converted = openai_request_to_company_payload(
        payload,
        defaults=GatewayDefaults(provider="google", locale="en-CN"),
    )

    assert converted["model"] == "gemini-3.5-flash"
    assert converted["provider"] == "google"
    assert converted["current_date"].endswith("Z")
    assert converted["messages"] == [
        {"author": "user", "content": {"text": "今天几号?"}}
    ]
    assert converted["tools"][0] == {
        "type": "local_tool",
        "function": {
            "name": "location-get-current-location",
            "description": "Gets current location.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_tool_conversion_never_outputs_remote_tool():
    converted = openai_request_to_company_payload(
        {
            "model": "google-gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "remote_tool", "name": "web_search"},
                {"name": "read_page"},
                {
                    "type": "local_tool",
                    "function": {
                        "name": "location-get-current-location",
                        "description": "Gets current location.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        defaults=GatewayDefaults(provider="google"),
    )

    assert converted["tools"] == [
        {
            "type": "local_tool",
            "function": {
                "name": "web_search",
                "description": "",
                "parameters": {},
            },
        },
        {
            "type": "local_tool",
            "function": {
                "name": "read_page",
                "description": "",
                "parameters": {},
            },
        },
        {
            "type": "local_tool",
            "function": {
                "name": "location-get-current-location",
                "description": "Gets current location.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def test_tool_result_messages_are_named_from_prior_tool_call():
    converted = openai_request_to_company_payload(
        {
            "model": "google-gemini-3.5-flash",
            "messages": [
                {"role": "user", "content": "今天几号？天气怎么样？"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "53681b26-50c2-44ab-925f-ec92f78e2e62",
                            "type": "function",
                            "function": {
                                "name": "location-get-current-location",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "53681b26-50c2-44ab-925f-ec92f78e2e62",
                    "content": '{"street":"新金桥路","latitude":31.245287128171167}',
                },
            ],
        },
        defaults=GatewayDefaults(provider="google"),
    )

    assert converted["messages"][1] == {
        "author": "assistant",
        "content": {"text": ""},
        "tool_calls": [
            {
                "id": "53681b26-50c2-44ab-925f-ec92f78e2e62",
                "type": "function",
                "function": {
                    "name": "location-get-current-location",
                    "arguments": {},
                },
            }
        ],
    }
    assert converted["messages"][2] == {
        "author": "tool",
        "content": {"text": '{"street":"新金桥路","latitude":31.245287128171167}'},
        "name": "location-get-current-location",
        "tool_call_id": "53681b26-50c2-44ab-925f-ec92f78e2e62",
    }


def test_openai_request_to_company_payload_accepts_raycast_model_ids():
    converted = openai_request_to_company_payload(
        {
            "model": "google-gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
        },
        defaults=GatewayDefaults(provider="google"),
    )

    assert converted["provider"] == "google"
    assert converted["model"] == "gemini-3.5-flash"


def test_openai_request_to_company_payload_uses_model_catalog_before_prefix_guessing():
    converted = openai_request_to_company_payload(
        {
            "model": "openai_o1-gpt-5",
            "messages": [{"role": "user", "content": "hi"}],
        },
        defaults=GatewayDefaults(provider="google"),
        model_catalog={
            "openai_o1-gpt-5": {
                "provider": "openai",
                "model": "gpt-5-from-catalog",
            }
        },
    )

    assert converted["provider"] == "openai"
    assert converted["model"] == "gpt-5-from-catalog"


def test_openai_request_to_company_payload_accepts_model_name_from_public_models_list():
    converted = openai_request_to_company_payload(
        {
            "model": "gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
        },
        defaults=GatewayDefaults(provider="openai"),
        model_catalog={
            "gemini-3.5-flash": {
                "provider": "google",
                "model": "gemini-3.5-flash",
                "reasoning_effort": "minimal",
            }
        },
    )

    assert converted["provider"] == "google"
    assert converted["model"] == "gemini-3.5-flash"
    assert converted["reasoning_effort"] == "minimal"


def test_openai_request_to_company_payload_uses_model_catalog_reasoning_effort_default():
    converted = openai_request_to_company_payload(
        {
            "model": "google-gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
        },
        defaults=GatewayDefaults(provider="google"),
        model_catalog={
            "google-gemini-3.5-flash": {
                "provider": "google",
                "model": "gemini-3.5-flash",
                "reasoning_effort": "minimal",
            }
        },
    )

    assert converted["reasoning_effort"] == "minimal"


def test_openai_request_reasoning_effort_overrides_catalog_default():
    converted = openai_request_to_company_payload(
        {
            "model": "google-gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        },
        defaults=GatewayDefaults(provider="google"),
        model_catalog={
            "google-gemini-3.5-flash": {
                "provider": "google",
                "model": "gemini-3.5-flash",
                "reasoning_effort": "minimal",
            }
        },
    )

    assert converted["reasoning_effort"] == "high"


def test_openai_request_accepts_responses_style_reasoning_effort():
    converted = openai_request_to_company_payload(
        {
            "model": "anthropic-claude-opus-4-8",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "high"},
        },
        defaults=GatewayDefaults(provider="google"),
        model_catalog={
            "anthropic-claude-opus-4-8": {
                "provider": "anthropic",
                "model": "claude-opus-4-8",
                "reasoning_effort": "none",
            }
        },
    )

    assert converted["provider"] == "anthropic"
    assert converted["model"] == "claude-opus-4-8"
    assert converted["reasoning_effort"] == "high"


def test_top_level_reasoning_effort_overrides_responses_style_reasoning_effort():
    converted = openai_request_to_company_payload(
        {
            "model": "anthropic-claude-opus-4-8",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "low",
            "reasoning": {"effort": "high"},
        },
        defaults=GatewayDefaults(provider="google"),
        model_catalog={
            "anthropic-claude-opus-4-8": {
                "provider": "anthropic",
                "model": "claude-opus-4-8",
                "reasoning_effort": "none",
            }
        },
    )

    assert converted["reasoning_effort"] == "low"


def test_responses_request_to_chat_payload_maps_input_and_instructions():
    converted = responses_request_to_chat_payload(
        {
            "model": "claude-opus-4-8",
            "instructions": "Answer in Chinese.",
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Be concise."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "今天几号？"}],
                },
            ],
            "reasoning": {"effort": "low"},
            "tools": [
                {
                    "type": "function",
                    "name": "location-get-current-location",
                    "description": "Gets current location.",
                    "parameters": {},
                }
            ],
        }
    )

    assert converted["messages"] == [{"role": "user", "content": "今天几号？"}]
    assert converted["additional_system_instructions"] == "Answer in Chinese.\n\nBe concise."
    assert converted["reasoning"] == {"effort": "low"}
    assert converted["tools"][0]["name"] == "location-get-current-location"


def test_responses_request_function_call_output_maps_to_tool_message():
    converted = responses_request_to_chat_payload(
        {
            "model": "gemini-3.5-flash",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "lookup",
                    "arguments": "{\"q\":\"weather\"}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "{\"city\":\"上海\"}",
                },
            ],
        }
    )

    assert converted["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"q\":\"weather\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "lookup",
            "content": "{\"city\":\"上海\"}",
        },
    ]


def test_responses_request_filters_partial_tool_call_history():
    converted = responses_request_to_chat_payload(
        {
            "model": "claude-opus-4-8",
            "input": [
                {"role": "user", "content": "read file"},
                {
                    "type": "function_call",
                    "call_id": "toolu_123",
                    "name": "exec_command",
                    "arguments": "",
                },
                {
                    "type": "function_call",
                    "call_id": "partial_1",
                    "name": "",
                    "arguments": "{\"cmd\":\"c",
                },
                {
                    "type": "function_call",
                    "call_id": "partial_2",
                    "name": "",
                    "arguments": "at 111.txt\"}",
                },
                {
                    "type": "function_call",
                    "call_id": "toolu_123",
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"cat 111.txt\"}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "partial_1",
                    "output": "unsupported call: ",
                },
                {
                    "type": "function_call_output",
                    "call_id": "toolu_123",
                    "output": "hello 333",
                },
            ],
        }
    )

    assert converted["messages"] == [
        {"role": "user", "content": "read file"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_123",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": "{\"cmd\":\"cat 111.txt\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_123",
            "name": "exec_command",
            "content": "hello 333",
        },
    ]


def test_responses_request_merges_assistant_text_with_tool_call_before_tool_result():
    converted = responses_request_to_chat_payload(
        {
            "model": "claude-opus-4-8",
            "input": [
                {"role": "user", "content": "read file"},
                {
                    "type": "function_call",
                    "call_id": "toolu_456",
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"printf hi\"}",
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Need to update the file.",
                        }
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "toolu_456",
                    "output": "hi",
                },
            ],
        }
    )

    assert converted["messages"] == [
        {"role": "user", "content": "read file"},
        {
            "role": "assistant",
            "content": "Need to update the file.",
            "tool_calls": [
                {
                    "id": "toolu_456",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": "{\"cmd\":\"printf hi\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_456",
            "name": "exec_command",
            "content": "hi",
        },
    ]


def test_responses_style_function_tools_are_local_tools_upstream():
    converted = openai_request_to_company_payload(
        responses_request_to_chat_payload(
            {
                "model": "gemini-3.5-flash",
                "input": "hi",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "description": "Lookup data.",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        ),
        defaults=GatewayDefaults(provider="google"),
    )

    assert converted["tools"] == [
        {
            "type": "local_tool",
            "function": {
                "name": "lookup",
                "description": "Lookup data.",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_split_model_for_company_handles_raycast_catalog_ids():
    assert split_model_for_company("openai-gpt-5.4-mini", "google") == (
        "gpt-5.4-mini",
        "openai",
    )
    assert split_model_for_company("openai_o1-o3", "google") == ("o3", "openai")
    assert split_model_for_company("groq-openai/gpt-oss-20b", "google") == (
        "openai/gpt-oss-20b",
        "groq",
    )
    assert split_model_for_company("baseten-moonshotai/Kimi-K2.6", "google") == (
        "moonshotai/Kimi-K2.6",
        "baseten",
    )


def test_raycast_model_catalog_maps_ids_to_provider_and_model():
    catalog = raycast_model_catalog(
        {
            "models": [
                {
                    "id": "google-gemini-3.5-flash",
                    "model": "gemini-3.5-flash",
                    "provider": "google",
                    "abilities": {
                        "reasoning_effort": {
                            "supported": True,
                            "options": ["minimal", "low", "medium", "high"],
                            "default": "minimal",
                        }
                    },
                },
                {
                    "id": "openai_o1-gpt-5",
                    "model": "gpt-5",
                    "provider": "openai",
                },
                {"id": "missing-provider", "model": "x"},
            ]
        }
    )

    assert catalog == {
        "google-gemini-3.5-flash": {
            "model": "gemini-3.5-flash",
            "provider": "google",
            "reasoning_effort": "minimal",
        },
        "gemini-3.5-flash": {
            "model": "gemini-3.5-flash",
            "provider": "google",
            "reasoning_effort": "minimal",
        },
        "openai_o1-gpt-5": {
            "model": "gpt-5",
            "provider": "openai",
        },
        "gpt-5": {
            "model": "gpt-5",
            "provider": "openai",
        },
    }


def test_resolve_model_for_company_prefers_catalog():
    assert resolve_model_for_company(
        "custom-id",
        "google",
        model_catalog={"custom-id": {"model": "real-model", "provider": "real-provider"}},
    ) == ("real-model", "real-provider")


def test_resolve_reasoning_effort_for_company_prefers_catalog():
    assert (
        resolve_reasoning_effort_for_company(
            "custom-id",
            model_catalog={
                "custom-id": {
                    "model": "real-model",
                    "provider": "real-provider",
                    "reasoning_effort": "xhigh",
                }
            },
        )
        == "xhigh"
    )
    assert resolve_reasoning_effort_for_company("missing", model_catalog={}) is None


def test_raycast_models_to_openai_models():
    converted = raycast_models_to_openai_models(
        {
            "models": [
                {
                    "id": "google-gemini-3.5-flash",
                    "model": "gemini-3.5-flash",
                    "provider": "google",
                    "name": "Gemini 3.5 Flash",
                },
                {"id": "openai-gpt-5.4-mini", "provider_name": "OpenAI"},
                {"provider": "missing-id"},
            ]
        },
        created=1780430484,
    )

    assert converted == {
        "object": "list",
        "data": [
            {
                "id": "gemini-3.5-flash",
                "object": "model",
                "created": 1780430484,
                "owned_by": "google",
            },
            {
                "id": "openai-gpt-5.4-mini",
                "object": "model",
                "created": 1780430484,
                "owned_by": "OpenAI",
            },
        ],
    }


def test_internal_reasoning_and_text_are_full_openai_like_stream_chunks():
    state = StreamState(
        request_id="chatcmpl_test",
        model="mimo-v2.5-pro",
        created=1780469337,
    )

    chunks = internal_chunk_to_openai_chunks(
        {
            "reasoning": "The user is asking",
            "text": "今天是 2026 年 6 月 2 日。",
        },
        state,
    )

    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"] == {
        "reasoning_content": "The user is asking"
    }
    assert chunks[2]["choices"][0]["delta"] == {
        "content": "今天是 2026 年 6 月 2 日。"
    }


def test_notification_chunks_are_ignored_until_visible_payload():
    state = StreamState(
        request_id="chatcmpl_test",
        model="model",
        created=1780469337,
        include_usage=True,
    )

    provider_status = internal_chunk_to_openai_chunks(
        {
            "notification_type": "provider_status",
            "notification": "",
            "status": "operational",
            "status_page_url": "https://status.claude.com",
            "text": "",
        },
        state,
    )
    ping = internal_chunk_to_openai_chunks(
        {"notification": "", "notification_type": "ping", "text": ""},
        state,
    )
    first_text = internal_chunk_to_openai_chunks({"text": "今"}, state)
    second_text = internal_chunk_to_openai_chunks({"text": "天是2026年6月2日。"}, state)
    finish = internal_chunk_to_openai_chunks(
        {
            "text": "",
            "finish_reason": "stop",
            "usage": {"input_tokens": 2, "output_tokens": 13},
        },
        state,
    )

    assert provider_status == []
    assert ping == []
    assert first_text[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert first_text[1]["choices"][0]["delta"] == {"content": "今"}
    assert second_text[0]["choices"][0]["delta"] == {"content": "天是2026年6月2日。"}
    assert finish[0]["choices"][0]["finish_reason"] == "stop"
    assert finish[1]["choices"] == []
    assert finish[1]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 13,
        "total_tokens": 15,
    }


def test_internal_tool_call_and_finish_are_split_into_valid_chunks():
    state = StreamState(
        request_id="chatcmpl_test",
        model="gemini-3.5-flash",
        created=1780469337,
    )

    chunks = internal_chunk_to_openai_chunks(
        {
            "finish_reason": "tool_calls",
            "text": "",
            "tool_calls": [
                {
                    "id": "f2b10fe1",
                    "name": "location-get-current-location",
                    "arguments": "{}",
                }
            ],
        },
        state,
    )

    assert chunks[1]["choices"][0]["delta"]["tool_calls"] == [
        {
            "index": 0,
            "id": "f2b10fe1",
            "type": "function",
            "function": {
                "name": "location-get-current-location",
                "arguments": "{}",
            },
        }
    ]
    assert chunks[2]["choices"][0]["finish_reason"] == "tool_calls"


def test_streaming_chat_tool_call_arguments_are_incremental():
    state = StreamState(
        request_id="chatcmpl_test",
        model="claude-opus-4-8",
        created=1780469337,
    )

    first = internal_chunk_to_openai_chunks(
        {
            "tool_calls": [
                {
                    "id": "toolu_123",
                    "index": 0,
                    "function": {"name": "exec_command", "arguments": ""},
                }
            ]
        },
        state,
    )
    second = internal_chunk_to_openai_chunks(
        {
            "tool_calls": [
                {
                    "id": "",
                    "index": 0,
                    "function": {"name": "", "arguments": "{\"cmd\":\"c"},
                }
            ]
        },
        state,
    )
    third = internal_chunk_to_openai_chunks(
        {
            "finish_reason": "tool_calls",
            "tool_calls": [
                {
                    "id": "toolu_123",
                    "index": 0,
                    "function": {
                        "name": "exec_command",
                        "arguments": "{\"cmd\":\"cat 111.txt\"}",
                    },
                }
            ],
        },
        state,
    )

    assert first[1]["choices"][0]["delta"]["tool_calls"] == [
        {
            "index": 0,
            "id": "toolu_123",
            "type": "function",
            "function": {"name": "exec_command"},
        }
    ]
    assert second[0]["choices"][0]["delta"]["tool_calls"][0]["function"] == {
        "arguments": "{\"cmd\":\"c"
    }
    assert third[0]["choices"][0]["finish_reason"] == "tool_calls"
    assert "tool_calls" not in third[0]["choices"][0]["delta"]


def test_usage_chunk_is_emitted_only_when_requested():
    state = StreamState(
        request_id="chatcmpl_test",
        model="gemini-3.5-flash",
        created=1780469337,
        include_usage=True,
    )

    chunks = internal_chunk_to_openai_chunks(
        {
            "finish_reason": "STOP",
            "usage": {"input_tokens": 1413, "output_tokens": 14},
        },
        state,
    )

    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 1413,
        "completion_tokens": 14,
        "total_tokens": 1427,
    }


def test_suspicious_stream_usage_uses_local_input_token_estimate():
    state = StreamState(
        request_id="chatcmpl_test",
        model="gemini-3.5-flash",
        created=1780469337,
        include_usage=True,
        input_token_estimate=120,
    )

    chunks = internal_chunk_to_openai_chunks(
        {
            "finish_reason": "STOP",
            "usage": {"input_tokens": 2, "output_tokens": 14},
        },
        state,
    )

    assert chunks[-1]["usage"] == {
        "prompt_tokens": 120,
        "completion_tokens": 14,
        "total_tokens": 134,
    }


def test_normal_stream_usage_keeps_upstream_input_tokens():
    state = StreamState(
        request_id="chatcmpl_test",
        model="gemini-3.5-flash",
        created=1780469337,
        include_usage=True,
        input_token_estimate=120,
    )

    chunks = internal_chunk_to_openai_chunks(
        {
            "finish_reason": "STOP",
            "usage": {"input_tokens": 1413, "output_tokens": 14},
        },
        state,
    )

    assert chunks[-1]["usage"] == {
        "prompt_tokens": 1413,
        "completion_tokens": 14,
        "total_tokens": 1427,
    }


def test_non_stream_response_aggregates_reasoning_extension():
    response = aggregate_openai_response(
        [
            {"reasoning": "Thinking. "},
            {"text": "Answer."},
            {"finish_reason": "STOP", "usage": {"input_tokens": 2, "output_tokens": 3}},
        ],
        request_id="chatcmpl_test",
        model="model",
        created=1,
        input_token_estimate=80,
    )

    message = response["choices"][0]["message"]
    assert response["object"] == "chat.completion"
    assert message["reasoning_content"] == "Thinking. "
    assert message["content"] == "Answer."
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"] == {
        "prompt_tokens": 80,
        "completion_tokens": 3,
        "total_tokens": 83,
    }


def test_non_stream_responses_response_aggregates_output_and_usage():
    response = aggregate_responses_response(
        [
            {"reasoning": "Thinking. "},
            {"text": "Answer."},
            {
                "finish_reason": "STOP",
                "usage": {"input_tokens": 2, "output_tokens": 3, "reasoning_tokens": 1},
            },
        ],
        request_id="resp_test",
        model="model",
        created=1,
        request_payload={
            "instructions": "Do not echo this.",
            "metadata": {"request": "metadata"},
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Large tool definition should not be echoed.",
                    "parameters": {"type": "object"},
                }
            ],
        },
        input_token_estimate=80,
    )

    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert response["output_text"] == "Answer."
    assert response["output"][0]["type"] == "message"
    assert response["output"][0]["content"][0]["text"] == "Answer."
    assert response["output"][1]["type"] == "reasoning"
    assert response["tool_choice"] == "auto"
    assert response["tools"] == []
    assert "instructions" not in response
    assert "metadata" not in response
    assert response["usage"] == {
        "input_tokens": 80,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": 83,
    }


def test_streaming_responses_events_emit_typed_sse_payloads():
    state = ResponsesStreamState(
        request_id="resp_test",
        model="model",
        created=1,
        input_token_estimate=50,
    )

    created = response_created_event(state)
    text_events = internal_chunk_to_response_events({"text": "今"}, state)
    finish_events = internal_chunk_to_response_events(
        {"finish_reason": "stop", "usage": {"input_tokens": 2, "output_tokens": 1}},
        state,
    )
    final_events = final_response_stream_events(
        state,
        request_payload={
            "instructions": "Do not echo this.",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Large tool definition should not be echoed.",
                    "parameters": {"type": "object"},
                }
            ],
        },
    )

    assert created["type"] == "response.created"
    assert created["response"]["output"] == []
    assert text_events[0]["type"] == "response.output_item.added"
    assert text_events[1]["type"] == "response.content_part.added"
    assert text_events[2]["type"] == "response.output_text.delta"
    assert text_events[2]["delta"] == "今"
    assert finish_events == []
    assert final_events[-1]["type"] == "response.completed"
    assert final_events[-1]["response"]["output_text"] == "今"
    assert final_events[-1]["response"]["tools"] == []
    assert "instructions" not in final_events[-1]["response"]
    assert final_events[-1]["response"]["usage"]["input_tokens"] == 50
    assert final_events[-1]["response"]["usage"]["total_tokens"] == 51
    assert encode_sse_event(text_events[2]).startswith("event: response.output_text.delta\n")


def test_estimate_input_tokens_uses_prompt_relevant_fields():
    estimate = estimate_input_tokens(
        {
            "model": "gemini-3.5-flash",
            "provider": "google",
            "thread_id": "not-counted",
            "messages": [{"author": "user", "content": {"text": "今天几号？天气怎么样？"}}],
            "tools": [
                {
                    "type": "local_tool",
                    "function": {
                        "name": "location-get-current-location",
                        "description": "Gets current location.",
                        "parameters": {},
                    },
                }
            ],
        }
    )

    assert estimate >= 8


def test_streaming_responses_tool_call_arguments_are_merged():
    state = ResponsesStreamState(request_id="resp_test", model="model", created=1)

    first = internal_chunk_to_response_events(
        {
            "tool_calls": [
                {
                    "id": "toolu_123",
                    "index": 0,
                    "function": {"name": "exec_command", "arguments": ""},
                }
            ]
        },
        state,
    )
    second = internal_chunk_to_response_events(
        {
            "tool_calls": [
                {
                    "id": "",
                    "index": 0,
                    "function": {"name": "", "arguments": "{\"cmd\":\"c"},
                }
            ]
        },
        state,
    )
    third = internal_chunk_to_response_events(
        {
            "finish_reason": "tool_calls",
            "tool_calls": [
                {
                    "id": "toolu_123",
                    "index": 0,
                    "function": {
                        "name": "exec_command",
                        "arguments": "{\"cmd\":\"cat 111.txt\"}",
                    },
                }
            ],
        },
        state,
    )

    assert [event["type"] for event in first] == ["response.output_item.added"]
    assert first[0]["item"]["call_id"] == "toolu_123"
    assert [event["type"] for event in second] == [
        "response.function_call_arguments.delta"
    ]
    assert second[0]["delta"] == "{\"cmd\":\"c"
    assert [event["type"] for event in third] == [
        "response.function_call_arguments.done",
        "response.output_item.done",
    ]
    assert third[0]["arguments"] == "{\"cmd\":\"cat 111.txt\"}"
    assert state.tool_calls == [
        {
            "id": third[1]["item"]["id"],
            "type": "function_call",
            "status": "completed",
            "call_id": "toolu_123",
            "name": "exec_command",
            "arguments": "{\"cmd\":\"cat 111.txt\"}",
        }
    ]


def test_normalize_responses_usage_accepts_chat_usage_names():
    assert normalize_responses_usage({"prompt_tokens": 2, "completion_tokens": 3}) == {
        "input_tokens": 2,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 5,
    }


def test_sse_helpers_parse_and_encode():
    parsed = company_sse_data_to_dict('data: {"text":"hi"}')

    assert parsed == {"text": "hi"}
    assert company_sse_data_to_dict("event: message") is None
    assert company_sse_data_to_dict("id: 123") is None
    assert company_sse_data_to_dict(": ping") is None
    assert company_sse_data_to_dict("data:") is None
    assert company_sse_data_to_dict("data: [DONE]") is None
    assert company_sse_data_to_dict("data: not-json") is None
    assert company_sse_data_to_dict("data: []") is None
    assert encode_sse_data({"text": "你好"}) == 'data: {"text":"你好"}\n\n'
    assert normalize_finish_reason("STOP") == "stop"
