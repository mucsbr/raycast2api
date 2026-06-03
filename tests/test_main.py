from raycast_gateway.adapters import GatewayDefaults, StreamState
from raycast_gateway.config import Settings
from raycast_gateway.main import (
    UpstreamSSEError,
    _debug_request_body,
    _log_request_summary,
    _redact_sensitive,
    _store_model_catalog,
    _stream_error_chunks,
    _upstream_http_exception,
)
from raycast_gateway.signing import RaycastSignatureConfig


def test_upstream_sse_context_error_maps_to_http_400():
    exc = UpstreamSSEError(
        {
            "error": {
                "message": "The message was too long (125%). Submit something shorter",
                "type": "context_error",
            }
        }
    )

    mapped = _upstream_http_exception(exc)

    assert mapped.status_code == 400
    assert mapped.detail["error"] == "context_error"
    assert mapped.detail["message"] == "The message was too long (125%). Submit something shorter"


def test_upstream_sse_error_streams_visible_openai_like_chunks():
    state = StreamState(request_id="chatcmpl_test", model="model", created=1)
    exc = UpstreamSSEError(
        {
            "error": {
                "message": "The message was too long (125%). Submit something shorter",
                "type": "context_error",
            }
        }
    )

    chunks = _stream_error_chunks(state, exc)

    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"] == {
        "content": "Upstream error (context_error): The message was too long (125%). Submit something shorter"
    }
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"


def test_store_model_catalog_uses_raycast_provider_and_model_fields():
    catalog = _store_model_catalog(
        {
            "models": [
                {
                    "id": "google-gemini-3.5-flash",
                    "model": "gemini-3.5-flash",
                    "provider": "google",
                }
            ]
        }
    )

    assert catalog["google-gemini-3.5-flash"] == {
        "model": "gemini-3.5-flash",
        "provider": "google",
    }


def test_log_request_summary_only_prints_model_fields(capsys):
    settings = Settings(
        company_api_url="",
        company_models_api_url="",
        company_api_key=None,
        request_timeout_seconds=1,
        log_request_summary=True,
        debug_request_body=False,
        debug_request_body_max_chars=4000,
        debug_upstream_stream=False,
        debug_upstream_max_chars=500,
        defaults=GatewayDefaults(),
        raycast_signature=RaycastSignatureConfig(),
    )

    _log_request_summary(
        settings,
        {
            "provider": "google",
            "model": "gemini-3.5-flash",
            "reasoning_effort": "minimal",
            "messages": [{"content": {"text": "do not print"}}],
        },
    )

    captured = capsys.readouterr()
    assert "provider=google model=gemini-3.5-flash reasoning_effort=minimal" in captured.err
    assert "do not print" not in captured.err


def test_debug_request_body_redacts_sensitive_fields(capsys):
    settings = Settings(
        company_api_url="",
        company_models_api_url="",
        company_api_key=None,
        request_timeout_seconds=1,
        log_request_summary=False,
        debug_request_body=True,
        debug_request_body_max_chars=4000,
        debug_upstream_stream=False,
        debug_upstream_max_chars=500,
        defaults=GatewayDefaults(),
        raycast_signature=RaycastSignatureConfig(),
    )

    _debug_request_body(
        settings,
        "client",
        {
            "model": "model",
            "reasoning": {"effort": "high"},
            "api_key": "dummy-api-key",
        },
    )

    captured = capsys.readouterr()
    assert "[raycast-gateway request-body:client]" in captured.err
    assert '"effort":"high"' in captured.err
    assert "dummy-api-key" not in captured.err
    assert "<redacted>" in captured.err


def test_redact_sensitive_is_recursive():
    assert _redact_sensitive(
        {
            "nested": [{"token": "abc"}, {"safe": "value"}],
            "X-Raycast-Signature-V2": "signature",
        }
    ) == {
        "nested": [{"token": "<redacted>"}, {"safe": "value"}],
        "X-Raycast-Signature-V2": "<redacted>",
    }
