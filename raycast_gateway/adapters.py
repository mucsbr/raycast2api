from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


OPENAI_DONE = "[DONE]"


@dataclass(frozen=True)
class GatewayDefaults:
    provider: str = "google"
    locale: str = "en-CN"
    source: str = "quick_ai"
    system_instruction: str = "markdown"
    debug: bool = False


@dataclass
class StreamState:
    request_id: str
    model: str
    created: int
    include_usage: bool = False
    role_sent: bool = False
    done_sent: bool = False
    tool_call_count: int = 0


@dataclass
class ResponsesStreamState:
    request_id: str
    model: str
    created: int
    include_usage: bool = False
    message_item_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    sequence_number: int = 0
    output_started: bool = False
    content_started: bool = False
    done_sent: bool = False
    text_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


def new_chat_completion_id() -> str:
    return f"chatcmpl_{uuid.uuid4().hex}"


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def utc_timestamp() -> int:
    return int(time.time())


def utc_iso_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def openai_request_to_company_payload(
    payload: dict[str, Any],
    *,
    defaults: GatewayDefaults | None = None,
    model_catalog: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    defaults = defaults or GatewayDefaults()
    model, provider = resolve_model_for_company(
        payload.get("model", ""),
        defaults.provider,
        model_catalog=model_catalog,
    )
    reasoning_effort = _reasoning_effort_from_payload(payload)
    if reasoning_effort is None:
        reasoning_effort = resolve_reasoning_effort_for_company(
            payload.get("model", ""),
            model_catalog=model_catalog,
        )

    company_payload: dict[str, Any] = {
        "current_date": payload.get("current_date") or utc_iso_timestamp(),
        "debug": payload.get("debug", defaults.debug),
        "locale": payload.get("locale", defaults.locale),
        "message_id": payload.get("message_id") or str(uuid.uuid4()).upper(),
        "messages": _messages_to_company(payload.get("messages", [])),
        "model": model,
        "provider": payload.get("provider", provider),
        "reasoning_effort": reasoning_effort,
        "source": payload.get("source", defaults.source),
        "system_instruction": payload.get("system_instruction", defaults.system_instruction),
        "thread_id": payload.get("thread_id") or str(uuid.uuid4()).upper(),
        "tool_choice": payload.get("tool_choice", "auto"),
        "tools": [_tool_to_company(tool) for tool in payload.get("tools", [])],
    }

    if "additional_system_instructions" in payload:
        company_payload["additional_system_instructions"] = payload["additional_system_instructions"]

    return {key: value for key, value in company_payload.items() if value is not None}


def responses_request_to_chat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    messages, instruction_parts = _responses_input_to_messages_and_instructions(
        payload.get("input", payload.get("messages", []))
    )

    instructions = _responses_content_to_text(payload.get("instructions"))
    if instructions:
        instruction_parts.insert(0, instructions)
    additional_system_instructions = _responses_content_to_text(
        payload.get("additional_system_instructions")
    )
    if additional_system_instructions:
        instruction_parts.append(additional_system_instructions)

    chat_payload: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": messages,
        "stream": payload.get("stream", False),
        "tool_choice": payload.get("tool_choice", "auto"),
        "tools": payload.get("tools", []),
    }

    passthrough_keys = (
        "current_date",
        "debug",
        "locale",
        "message_id",
        "provider",
        "reasoning",
        "reasoning_effort",
        "source",
        "system_instruction",
        "temperature",
        "thread_id",
    )
    for key in passthrough_keys:
        if key in payload:
            chat_payload[key] = payload[key]

    if instruction_parts:
        chat_payload["additional_system_instructions"] = "\n\n".join(
            part for part in instruction_parts if part
        )

    return {key: value for key, value in chat_payload.items() if value is not None}


def internal_chunk_to_openai_chunks(
    internal: dict[str, Any],
    state: StreamState,
) -> list[dict[str, Any]]:
    if not _has_openai_visible_payload(internal):
        return []

    chunks: list[dict[str, Any]] = []

    if not state.role_sent:
        chunks.append(_stream_chunk(state, {"role": "assistant"}))
        state.role_sent = True

    reasoning = internal.get("reasoning")
    if reasoning:
        chunks.append(_stream_chunk(state, {"reasoning_content": str(reasoning)}))

    text = internal.get("text")
    if text:
        chunks.append(_stream_chunk(state, {"content": str(text)}))

    tool_calls = internal.get("tool_calls") or []
    if tool_calls:
        chunks.append(_stream_chunk(state, {"tool_calls": _tool_calls_to_openai(tool_calls, state)}))

    finish_reason = normalize_finish_reason(internal.get("finish_reason"))
    if finish_reason:
        chunks.append(_stream_chunk(state, {}, finish_reason=finish_reason))

    usage = internal.get("usage")
    if usage and state.include_usage:
        chunks.append(_usage_chunk(state, usage))

    return chunks


def response_created_event(
    state: ResponsesStreamState,
    *,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _response_event(
        state,
        "response.created",
        response=_response_object_from_parts(
            request_id=state.request_id,
            model=state.model,
            created=state.created,
            status="in_progress",
            output_text="",
            reasoning_text="",
            tool_calls=[],
            usage=None,
            request_payload=request_payload,
            include_empty_message=False,
        ),
    )


def internal_chunk_to_response_events(
    internal: dict[str, Any],
    state: ResponsesStreamState,
) -> list[dict[str, Any]]:
    if not _has_openai_visible_payload(internal):
        return []

    events: list[dict[str, Any]] = []

    reasoning = internal.get("reasoning")
    if reasoning:
        reasoning_text = str(reasoning)
        state.reasoning_parts.append(reasoning_text)
        events.append(
            _response_event(
                state,
                "response.reasoning_text.delta",
                response_id=state.request_id,
                item_id=f"rs_{state.request_id.removeprefix('resp_')}",
                output_index=1 if state.output_started else 0,
                content_index=0,
                delta=reasoning_text,
            )
        )

    text = internal.get("text")
    if text:
        text_delta = str(text)
        events.extend(_ensure_response_text_output_events(state))
        state.text_parts.append(text_delta)
        events.append(
            _response_event(
                state,
                "response.output_text.delta",
                response_id=state.request_id,
                item_id=state.message_item_id,
                output_index=0,
                content_index=0,
                delta=text_delta,
            )
        )

    tool_calls = internal.get("tool_calls") or []
    if tool_calls:
        for tool_call in _tool_calls_to_response_items(tool_calls):
            output_index = (1 if state.output_started else 0) + len(state.tool_calls)
            state.tool_calls.append(tool_call)
            events.append(
                _response_event(
                    state,
                    "response.output_item.added",
                    response_id=state.request_id,
                    output_index=output_index,
                    item={**tool_call, "status": "in_progress"},
                )
            )
            events.append(
                _response_event(
                    state,
                    "response.output_item.done",
                    response_id=state.request_id,
                    output_index=output_index,
                    item=tool_call,
                )
            )

    finish_reason = normalize_finish_reason(internal.get("finish_reason"))
    if finish_reason:
        state.finish_reason = finish_reason

    usage = internal.get("usage")
    if usage:
        state.usage = usage

    return events


def final_response_stream_events(
    state: ResponsesStreamState,
    *,
    request_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if state.done_sent:
        return []
    state.done_sent = True

    events: list[dict[str, Any]] = []
    output_text = "".join(state.text_parts)

    if state.content_started:
        content_part = {
            "type": "output_text",
            "text": output_text,
            "annotations": [],
        }
        events.append(
            _response_event(
                state,
                "response.output_text.done",
                response_id=state.request_id,
                item_id=state.message_item_id,
                output_index=0,
                content_index=0,
                text=output_text,
            )
        )
        events.append(
            _response_event(
                state,
                "response.content_part.done",
                response_id=state.request_id,
                item_id=state.message_item_id,
                output_index=0,
                content_index=0,
                part=content_part,
            )
        )
        events.append(
            _response_event(
                state,
                "response.output_item.done",
                response_id=state.request_id,
                output_index=0,
                item=_response_message_item(state.message_item_id, output_text),
            )
        )

    events.append(
        _response_event(
            state,
            "response.completed",
            response=_response_object_from_parts(
                request_id=state.request_id,
                model=state.model,
                created=state.created,
                status="completed",
                output_text=output_text,
                reasoning_text="".join(state.reasoning_parts),
                tool_calls=state.tool_calls,
                usage=state.usage,
                request_payload=request_payload,
            ),
        )
    )
    return events


def response_failed_event(
    state: ResponsesStreamState,
    *,
    error: dict[str, Any],
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state.done_sent = True
    return _response_event(
        state,
        "response.failed",
        response=_response_object_from_parts(
            request_id=state.request_id,
            model=state.model,
            created=state.created,
            status="failed",
            output_text="".join(state.text_parts),
            reasoning_text="".join(state.reasoning_parts),
            tool_calls=state.tool_calls,
            usage=state.usage,
            request_payload=request_payload,
            error=error,
            include_empty_message=False,
        ),
    )


def _has_openai_visible_payload(internal: dict[str, Any]) -> bool:
    return bool(
        internal.get("reasoning")
        or internal.get("text")
        or internal.get("tool_calls")
        or internal.get("finish_reason")
        or internal.get("usage")
    )


def final_openai_stream_chunks(state: StreamState) -> list[dict[str, Any]]:
    if state.done_sent:
        return []
    state.done_sent = True
    if state.role_sent:
        return []
    state.role_sent = True
    return [_stream_chunk(state, {"role": "assistant"})]


def company_sse_data_to_dict(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None

    if not stripped.startswith("data:"):
        return None

    stripped = stripped[5:].strip()
    if not stripped or stripped == OPENAI_DONE:
        return None

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    return data


def encode_sse_data(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        body = payload
    else:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {body}\n\n"


def encode_sse_event(payload: dict[str, Any]) -> str:
    event_type = str(payload.get("type") or "message")
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {body}\n\n"


def aggregate_openai_response(
    internal_chunks: list[dict[str, Any]],
    *,
    request_id: str,
    model: str,
    created: int,
) -> dict[str, Any]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    state = StreamState(request_id=request_id, model=model, created=created)

    for internal in internal_chunks:
        if internal.get("text"):
            content_parts.append(str(internal["text"]))
        if internal.get("reasoning"):
            reasoning_parts.append(str(internal["reasoning"]))
        if internal.get("tool_calls"):
            tool_calls.extend(_tool_calls_to_openai(internal["tool_calls"], state))
        if internal.get("finish_reason"):
            finish_reason = normalize_finish_reason(internal["finish_reason"])
        if internal.get("usage"):
            usage = normalize_usage(internal["usage"])

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) if content_parts else None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or "stop",
            }
        ],
    }
    if usage:
        response["usage"] = usage
    return response


def aggregate_responses_response(
    internal_chunks: list[dict[str, Any]],
    *,
    request_id: str,
    model: str,
    created: int,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None

    for internal in internal_chunks:
        if internal.get("text"):
            content_parts.append(str(internal["text"]))
        if internal.get("reasoning"):
            reasoning_parts.append(str(internal["reasoning"]))
        if internal.get("tool_calls"):
            tool_calls.extend(_tool_calls_to_response_items(internal["tool_calls"]))
        if internal.get("usage"):
            usage = internal["usage"]

    return _response_object_from_parts(
        request_id=request_id,
        model=model,
        created=created,
        status="completed",
        output_text="".join(content_parts),
        reasoning_text="".join(reasoning_parts),
        tool_calls=tool_calls,
        usage=usage,
        request_payload=request_payload,
    )


def raycast_models_to_openai_models(payload: dict[str, Any], *, created: int = 0) -> dict[str, Any]:
    models = payload.get("models", [])
    data: list[dict[str, Any]] = []

    if not isinstance(models, list):
        models = []

    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("model") or model.get("id")
        if not model_id:
            continue
        data.append(
            {
                "id": str(model_id),
                "object": "model",
                "created": created,
                "owned_by": str(model.get("provider") or model.get("provider_name") or "raycast"),
            }
        )

    return {
        "object": "list",
        "data": data,
    }


def raycast_model_catalog(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    models = payload.get("models", [])
    catalog: dict[str, dict[str, str]] = {}

    if not isinstance(models, list):
        return catalog

    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        model_name = model.get("model")
        provider = model.get("provider")
        if not model_id or not model_name or not provider:
            continue
        entry = {
            "model": str(model_name),
            "provider": str(provider),
        }
        reasoning_effort = _default_reasoning_effort(model)
        if reasoning_effort is not None:
            entry["reasoning_effort"] = reasoning_effort

        catalog[str(model_id)] = entry
        catalog[str(model_name)] = entry

    return catalog


def resolve_model_for_company(
    model: str,
    default_provider: str,
    *,
    model_catalog: dict[str, dict[str, str]] | None = None,
) -> tuple[str, str]:
    if model_catalog and model in model_catalog:
        entry = model_catalog[model]
        return entry["model"], entry["provider"]
    return split_model_for_company(model, default_provider)


def resolve_reasoning_effort_for_company(
    model: str,
    *,
    model_catalog: dict[str, dict[str, str]] | None = None,
) -> str | None:
    if not model_catalog or model not in model_catalog:
        return None
    return model_catalog[model].get("reasoning_effort")


def _reasoning_effort_from_payload(payload: dict[str, Any]) -> Any:
    if payload.get("reasoning_effort") is not None:
        return payload["reasoning_effort"]

    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        return reasoning["effort"]

    return None


def split_model_for_company(model: str, default_provider: str) -> tuple[str, str]:
    if "/" in model and not model.startswith(("groq-", "baseten-", "together-")):
        provider, model_name = model.split("/", 1)
        if provider and model_name:
            return model_name, provider

    raycast_prefixes = {
        "anthropic": "anthropic",
        "baseten": "baseten",
        "google": "google",
        "groq": "groq",
        "mistral": "mistral",
        "openai": "openai",
        "openai_o1": "openai",
        "perplexity": "perplexity",
        "raycast": "raycast",
        "together": "together",
        "xai": "xai",
    }
    for prefix, provider in sorted(raycast_prefixes.items(), key=lambda item: len(item[0]), reverse=True):
        marker = f"{prefix}-"
        if model.startswith(marker):
            return model[len(marker):], provider

    return model, default_provider


def _default_reasoning_effort(model: dict[str, Any]) -> str | None:
    abilities = model.get("abilities")
    if not isinstance(abilities, dict):
        return None
    reasoning_effort = abilities.get("reasoning_effort")
    if not isinstance(reasoning_effort, dict):
        return None
    default = reasoning_effort.get("default")
    if default is None:
        return None
    return str(default)


def normalize_finish_reason(reason: Any) -> str | None:
    if reason is None:
        return None
    normalized = str(reason).strip()
    if not normalized:
        return None
    mapping = {
        "STOP": "stop",
        "stop": "stop",
        "TOOL_CALLS": "tool_calls",
        "tool_calls": "tool_calls",
        "MAX_TOKENS": "length",
        "length": "length",
    }
    return mapping.get(normalized, normalized.lower())


def normalize_usage(usage: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    normalized: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if "reasoning_tokens" in usage:
        normalized["completion_tokens_details"] = {
            "reasoning_tokens": int(usage["reasoning_tokens"] or 0)
        }
    return normalized


def normalize_responses_usage(usage: dict[str, Any]) -> dict[str, Any]:
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {"cached_tokens": int(usage.get("cached_tokens", 0) or 0)}
    if not isinstance(output_details, dict):
        output_details = {
            "reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0)
        }
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": input_details,
        "output_tokens": output_tokens,
        "output_tokens_details": output_details,
        "total_tokens": total_tokens,
    }


def _responses_input_to_messages_and_instructions(
    input_value: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}], []

    if not isinstance(input_value, list):
        return [], []

    messages: list[dict[str, Any]] = []
    instruction_parts: list[str] = []
    for item in input_value:
        item_messages, item_instructions = _responses_item_to_messages_and_instructions(item)
        messages.extend(item_messages)
        instruction_parts.extend(item_instructions)

    return messages, instruction_parts


def _responses_item_to_messages_and_instructions(
    item: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(item, str):
        return [{"role": "user", "content": item}], []
    if not isinstance(item, dict):
        return [{"role": "user", "content": _responses_content_to_text(item)}], []

    item_type = item.get("type")
    if item_type == "function_call":
        call_id = str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}")
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments", ""),
                        },
                    }
                ],
            }
        ], []

    if item_type == "function_call_output":
        call_id = item.get("call_id") or item.get("id")
        return [
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _responses_content_to_text(
                    item.get("output", item.get("content", ""))
                ),
            }
        ], []

    if item_type in {"input_text", "output_text"}:
        return [{"role": "user", "content": _responses_content_to_text(item)}], []

    if item_type in {"reasoning", "item_reference"}:
        return [], []

    role = str(item.get("role") or "user")
    text = _responses_content_to_text(item.get("content", item.get("text", "")))
    if role in {"developer", "system"}:
        return [], [text] if text else []
    return [{"role": role, "content": text}], []


def _responses_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return _responses_content_to_text(content["content"])
        if "output" in content:
            return _responses_content_to_text(content["output"])
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _responses_content_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(content)


def _messages_to_company(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_call_names = _collect_tool_call_names(messages)
    return [_message_to_company(message, tool_call_names) for message in messages]


def _collect_tool_call_names(messages: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            tool_call_id = tool_call.get("id")
            function = tool_call.get("function") or {}
            name = tool_call.get("name") or function.get("name")
            if tool_call_id and name:
                names[str(tool_call_id)] = str(name)
    return names


def _message_to_company(
    message: dict[str, Any],
    tool_call_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    tool_call_names = tool_call_names or {}
    converted: dict[str, Any] = {
        "author": message.get("role", "user"),
        "content": {"text": _content_to_text(message.get("content"))},
    }

    tool_call_id = message.get("tool_call_id")
    name = message.get("name")
    if not name and tool_call_id:
        name = tool_call_names.get(str(tool_call_id))

    if name:
        converted["name"] = name
    if tool_call_id:
        converted["tool_call_id"] = tool_call_id
    if message.get("tool_calls"):
        converted["tool_calls"] = [_message_tool_call_to_company(tool_call) for tool_call in message["tool_calls"]]
    return converted


def _message_tool_call_to_company(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    return {
        "id": tool_call.get("id"),
        "type": tool_call.get("type", "function"),
        "function": {
            "name": tool_call.get("name") or function.get("name"),
            "arguments": _arguments_to_object(function.get("arguments", tool_call.get("arguments", {}))),
        },
    }


def _arguments_to_object(arguments: Any) -> Any:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {
                "text",
                "input_text",
                "output_text",
                "summary_text",
                "reasoning_text",
            }:
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _tool_to_company(tool: dict[str, Any]) -> dict[str, Any]:
    tool_type = tool.get("type")
    if tool_type in {"function", "local_tool"}:
        return _function_tool_to_local_tool(tool.get("function") or tool)
    if tool_type == "remote_tool":
        return _function_tool_to_local_tool(tool)
    if "name" in tool:
        return _function_tool_to_local_tool(tool)
    if tool_type:
        return _function_tool_to_local_tool(
            {
                "name": tool_type,
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
        )
    return tool


def _function_tool_to_local_tool(function: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "local_tool",
        "function": {
            "name": function.get("name"),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        },
    }


def _tool_calls_to_openai(tool_calls: list[dict[str, Any]], state: StreamState) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        index = state.tool_call_count
        state.tool_call_count += 1
        converted.append(
            {
                "index": index,
                "id": tool_call.get("id") or f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": tool_call.get("name")
                    or tool_call.get("function", {}).get("name"),
                    "arguments": _arguments_to_string(
                        tool_call.get("arguments")
                        if "arguments" in tool_call
                        else tool_call.get("function", {}).get("arguments", "")
                    ),
                },
            }
        )
    return converted


def _arguments_to_string(arguments: Any) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _ensure_response_text_output_events(state: ResponsesStreamState) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not state.output_started:
        state.output_started = True
        events.append(
            _response_event(
                state,
                "response.output_item.added",
                response_id=state.request_id,
                output_index=0,
                item={**_response_message_item(state.message_item_id, ""), "status": "in_progress"},
            )
        )
    if not state.content_started:
        state.content_started = True
        events.append(
            _response_event(
                state,
                "response.content_part.added",
                response_id=state.request_id,
                item_id=state.message_item_id,
                output_index=0,
                content_index=0,
                part={"type": "output_text", "text": "", "annotations": []},
            )
        )
    return events


def _response_event(
    state: ResponsesStreamState,
    event_type: str,
    **fields: Any,
) -> dict[str, Any]:
    event = {
        "type": event_type,
        "sequence_number": state.sequence_number,
    }
    state.sequence_number += 1
    event.update(fields)
    return event


def _response_object_from_parts(
    *,
    request_id: str,
    model: str,
    created: int,
    status: str,
    output_text: str,
    reasoning_text: str,
    tool_calls: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    request_payload: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    include_empty_message: bool = True,
) -> dict[str, Any]:
    request_payload = request_payload or {}
    output = _response_output_items(
        output_text,
        reasoning_text,
        tool_calls,
        include_empty_message=include_empty_message,
    )
    response: dict[str, Any] = {
        "id": request_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": error,
        "incomplete_details": None,
        "model": model,
        "output": output,
        "output_text": output_text,
        "parallel_tool_calls": request_payload.get("parallel_tool_calls", True),
        "previous_response_id": request_payload.get("previous_response_id"),
        "store": request_payload.get("store", False),
        "tool_choice": request_payload.get("tool_choice", "auto"),
        "tools": request_payload.get("tools", []),
        "usage": normalize_responses_usage(usage) if usage else None,
    }

    optional_keys = (
        "instructions",
        "max_output_tokens",
        "metadata",
        "reasoning",
        "temperature",
        "text",
        "top_p",
        "truncation",
        "user",
    )
    for key in optional_keys:
        if key in request_payload:
            response[key] = request_payload[key]

    return response


def _response_output_items(
    output_text: str,
    reasoning_text: str,
    tool_calls: list[dict[str, Any]],
    *,
    include_empty_message: bool = True,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if output_text or (include_empty_message and not tool_calls):
        output.append(_response_message_item(f"msg_{uuid.uuid4().hex}", output_text))
    if reasoning_text:
        output.append(
            {
                "id": f"rs_{uuid.uuid4().hex}",
                "type": "reasoning",
                "status": "completed",
                "summary": [],
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": reasoning_text,
                    }
                ],
            }
        )
    output.extend(tool_calls)
    return output


def _response_message_item(item_id: str, output_text: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": output_text,
                "annotations": [],
            }
        ],
    }


def _tool_calls_to_response_items(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_tool_call_to_response_item(tool_call) for tool_call in tool_calls]


def _tool_call_to_response_item(tool_call: dict[str, Any]) -> dict[str, Any]:
    call_id = str(tool_call.get("id") or tool_call.get("call_id") or f"call_{uuid.uuid4().hex}")
    function = tool_call.get("function") or {}
    return {
        "id": tool_call.get("response_id") or f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": tool_call.get("name") or function.get("name"),
        "arguments": _arguments_to_string(
            tool_call.get("arguments")
            if "arguments" in tool_call
            else function.get("arguments", "")
        ),
    }


def _stream_chunk(
    state: StreamState,
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": state.request_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _usage_chunk(state: StreamState, usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": state.request_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        "choices": [],
        "usage": normalize_usage(usage),
    }
