from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .adapters import (
    OPENAI_DONE,
    StreamState,
    aggregate_openai_response,
    company_sse_data_to_dict,
    encode_sse_data,
    final_openai_stream_chunks,
    internal_chunk_to_openai_chunks,
    new_chat_completion_id,
    openai_request_to_company_payload,
    raycast_model_catalog,
    raycast_models_to_openai_models,
    utc_timestamp,
)
from .config import Settings, get_settings
from .signing import build_raycast_signature_headers, serialize_json_body

app = FastAPI(title="Raycast Gateway", version="0.1.0")

MODEL_CATALOG_TTL_SECONDS = 300
_model_catalog_cache: dict[str, dict[str, str]] = {}
_model_catalog_cached_at = 0.0


class UpstreamSSEError(Exception):
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "upstream stream error")
            self.error_type = str(error.get("type") or "upstream_error")
        else:
            message = "upstream stream error"
            self.error_type = "upstream_error"
        super().__init__(message)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(request: Request) -> Response:
    payload = await request.json()
    settings = get_settings()
    _debug_request_body(settings, "client", payload)
    if not settings.company_api_url:
        raise HTTPException(
            status_code=500,
            detail="COMPANY_API_URL is not configured",
        )

    try:
        model_catalog = await _get_model_catalog(settings)
    except httpx.HTTPError:
        model_catalog = {}

    company_payload = openai_request_to_company_payload(
        payload,
        defaults=settings.defaults,
        model_catalog=model_catalog,
    )
    _log_request_summary(settings, company_payload)
    _debug_request_body(settings, "company", company_payload)
    try:
        upstream_body, upstream_headers = _build_upstream_request(company_payload, settings)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    request_id = new_chat_completion_id()
    created = utc_timestamp()
    model = payload.get("model") or company_payload.get("model", "")
    include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

    if payload.get("stream", False):
        state = StreamState(
            request_id=request_id,
            model=model,
            created=created,
            include_usage=include_usage,
        )
        return StreamingResponse(
            _stream_company_as_openai(upstream_body, upstream_headers, state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        internal_chunks = await _collect_company_response(upstream_body, upstream_headers)
    except (httpx.HTTPError, UpstreamSSEError) as exc:
        raise _upstream_http_exception(exc) from exc

    return JSONResponse(
        aggregate_openai_response(
            internal_chunks,
            request_id=request_id,
            model=model,
            created=created,
        )
    )


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    settings = get_settings()
    if not settings.company_models_api_url:
        raise HTTPException(
            status_code=500,
            detail="COMPANY_MODELS_API_URL is not configured",
        )

    try:
        headers = _build_upstream_get_headers(settings, signature_body="{}")
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        data = await _get_upstream_json(settings.company_models_api_url, headers, settings)
    except httpx.HTTPError as exc:
        raise _upstream_http_exception(exc) from exc

    _store_model_catalog(data)
    return JSONResponse(raycast_models_to_openai_models(data))


async def _stream_company_as_openai(
    upstream_body: bytes,
    upstream_headers: dict[str, str],
    state: StreamState,
) -> AsyncIterator[str]:
    try:
        async for internal in _iter_company_stream(upstream_body, upstream_headers):
            for chunk in internal_chunk_to_openai_chunks(internal, state):
                yield encode_sse_data(chunk)
    except (httpx.HTTPError, UpstreamSSEError) as exc:
        for chunk in _stream_error_chunks(state, exc):
            yield encode_sse_data(chunk)

    for chunk in final_openai_stream_chunks(state):
        yield encode_sse_data(chunk)
    yield encode_sse_data(OPENAI_DONE)


async def _collect_company_response(
    upstream_body: bytes,
    upstream_headers: dict[str, str],
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    async for internal in _iter_company_stream(upstream_body, upstream_headers):
        chunks.append(internal)
    return chunks


async def _iter_company_stream(
    upstream_body: bytes,
    upstream_headers: dict[str, str],
) -> AsyncIterator[dict[str, Any]]:
    settings = get_settings()
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            settings.company_api_url,
            content=upstream_body,
            headers=upstream_headers,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            _debug_upstream_line(settings, f"status={response.status_code} content_type={content_type}")
            if "text/event-stream" not in content_type:
                await response.aread()
                _debug_upstream_line(settings, response.text)
                data = response.json()
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            yield item
                    return
                if isinstance(data, dict):
                    yield data
                    return

            current_event: str | None = None
            async for line in response.aiter_lines():
                _debug_upstream_line(settings, line)
                stripped = line.strip()
                if stripped.startswith("event:"):
                    current_event = stripped[6:].strip()
                    continue
                if not stripped:
                    current_event = None
                    continue
                internal = company_sse_data_to_dict(line)
                if internal is not None:
                    if current_event == "error" or "error" in internal:
                        raise UpstreamSSEError(internal)
                    yield internal


def _build_upstream_request(
    company_payload: dict[str, Any],
    settings: Settings,
) -> tuple[bytes, dict[str, str]]:
    serialized_body = serialize_json_body(company_payload)
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
    }
    if settings.company_api_key:
        headers["Authorization"] = f"Bearer {settings.company_api_key}"

    headers.update(
        build_raycast_signature_headers(
            serialized_body,
            settings.raycast_signature,
        )
    )

    if settings.raycast_signature.accept_language:
        headers["Accept-Language"] = settings.raycast_signature.accept_language
    if settings.raycast_signature.experimental:
        headers["X-Raycast-Experimental"] = settings.raycast_signature.experimental
    if settings.raycast_signature.user_agent:
        headers["User-Agent"] = settings.raycast_signature.user_agent
    if settings.raycast_signature.priority:
        headers["Priority"] = settings.raycast_signature.priority

    return serialized_body.encode("utf-8"), headers


def _build_upstream_get_headers(
    settings: Settings,
    *,
    signature_body: str = "",
) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if settings.company_api_key:
        headers["Authorization"] = f"Bearer {settings.company_api_key}"

    headers.update(
        build_raycast_signature_headers(
            signature_body,
            settings.raycast_signature,
        )
    )

    if settings.raycast_signature.accept_language:
        headers["Accept-Language"] = settings.raycast_signature.accept_language
    if settings.raycast_signature.experimental:
        headers["X-Raycast-Experimental"] = settings.raycast_signature.experimental
    if settings.raycast_signature.user_agent:
        headers["User-Agent"] = settings.raycast_signature.user_agent
    if settings.raycast_signature.priority:
        headers["Priority"] = settings.raycast_signature.priority

    return headers


async def _get_upstream_json(
    url: str,
    headers: dict[str, str],
    settings: Settings,
) -> dict[str, Any]:
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    last_error: httpx.HTTPError | None = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(2):
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise httpx.DecodingError("upstream returned a non-object JSON response")
                return data
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(0.3)
                    continue
                raise

    if last_error:
        raise last_error
    raise httpx.TransportError("upstream request failed")


async def _get_model_catalog(settings: Settings) -> dict[str, dict[str, str]]:
    global _model_catalog_cached_at

    if _model_catalog_cache and time.time() - _model_catalog_cached_at < MODEL_CATALOG_TTL_SECONDS:
        return _model_catalog_cache
    if not settings.company_models_api_url:
        return {}

    headers = _build_upstream_get_headers(settings, signature_body="{}")
    data = await _get_upstream_json(settings.company_models_api_url, headers, settings)
    return _store_model_catalog(data)


def _store_model_catalog(data: dict[str, Any]) -> dict[str, dict[str, str]]:
    global _model_catalog_cache, _model_catalog_cached_at

    _model_catalog_cache = raycast_model_catalog(data)
    _model_catalog_cached_at = time.time()
    return _model_catalog_cache


def _upstream_http_exception(exc: httpx.HTTPError | UpstreamSSEError) -> HTTPException:
    if isinstance(exc, UpstreamSSEError):
        return HTTPException(
            status_code=400 if exc.error_type == "context_error" else 502,
            detail={
                "error": exc.error_type,
                "message": str(exc),
                "upstream": exc.payload,
            },
        )

    detail: dict[str, Any] = {
        "error": "upstream_request_failed",
        "message": str(exc) or exc.__class__.__name__,
    }
    if isinstance(exc, httpx.HTTPStatusError):
        detail["upstream_status"] = exc.response.status_code
        detail["upstream_body"] = exc.response.text[:500]
    return HTTPException(status_code=502, detail=detail)


def _stream_error_chunks(state: StreamState, exc: httpx.HTTPError | UpstreamSSEError) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if not state.role_sent:
        chunks.append(_openai_stream_chunk(state, {"role": "assistant"}))
        state.role_sent = True

    chunks.append(_stream_error_chunk(state, exc))
    chunks.append(_openai_stream_chunk(state, {}, finish_reason="stop"))
    return chunks


def _stream_error_chunk(state: StreamState, exc: httpx.HTTPError | UpstreamSSEError) -> dict[str, Any]:
    if isinstance(exc, UpstreamSSEError):
        message = f"Upstream error ({exc.error_type}): {str(exc)}"
    else:
        message = f"Upstream request failed: {str(exc) or exc.__class__.__name__}"
    return _openai_stream_chunk(state, {"content": message})


def _openai_stream_chunk(
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


def _debug_upstream_line(settings: Settings, value: str) -> None:
    if not settings.debug_upstream_stream:
        return

    max_chars = max(settings.debug_upstream_max_chars, 0)
    rendered = value if len(value) <= max_chars else f"{value[:max_chars]}...<truncated>"
    print(f"[raycast-gateway upstream] {rendered}", file=sys.stderr, flush=True)


def _log_request_summary(settings: Settings, company_payload: dict[str, Any]) -> None:
    if not settings.log_request_summary:
        return

    print(
        "[raycast-gateway request] "
        f"provider={company_payload.get('provider')} "
        f"model={company_payload.get('model')} "
        f"reasoning_effort={company_payload.get('reasoning_effort')}",
        file=sys.stderr,
        flush=True,
    )


def _debug_request_body(settings: Settings, label: str, payload: dict[str, Any]) -> None:
    if not settings.debug_request_body:
        return

    body = json.dumps(_redact_sensitive(payload), ensure_ascii=False, separators=(",", ":"))
    max_chars = max(settings.debug_request_body_max_chars, 0)
    rendered = body if len(body) <= max_chars else f"{body[:max_chars]}...<truncated>"
    print(f"[raycast-gateway request-body:{label}] {rendered}", file=sys.stderr, flush=True)


def _redact_sensitive(value: Any) -> Any:
    sensitive_keys = {
        "authorization",
        "api_key",
        "apikey",
        "bearer",
        "company_api_key",
        "jwt_secret",
        "raycast_signing_secret",
        "secret",
        "signature",
        "token",
        "x-raycast-signature",
        "x-raycast-signature-v2",
    }

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in sensitive_keys:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact_sensitive(item)
        return redacted

    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]

    return value
