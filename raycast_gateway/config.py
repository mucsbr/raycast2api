from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .adapters import GatewayDefaults
from .signing import RaycastSignatureConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    company_api_url: str
    company_models_api_url: str
    company_api_key: str | None
    request_timeout_seconds: float
    log_request_summary: bool
    debug_request_body: bool
    debug_request_body_max_chars: int
    debug_upstream_stream: bool
    debug_upstream_max_chars: int
    defaults: GatewayDefaults
    raycast_signature: RaycastSignatureConfig


def get_settings() -> Settings:
    load_env_file()
    return Settings(
        company_api_url=os.environ.get("COMPANY_API_URL", ""),
        company_models_api_url=os.environ.get("COMPANY_MODELS_API_URL", ""),
        company_api_key=os.environ.get("COMPANY_API_KEY") or None,
        request_timeout_seconds=float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "120")),
        log_request_summary=_env_bool("LOG_REQUEST_SUMMARY", False),
        debug_request_body=_env_bool("DEBUG_REQUEST_BODY", False),
        debug_request_body_max_chars=int(os.environ.get("DEBUG_REQUEST_BODY_MAX_CHARS", "4000")),
        debug_upstream_stream=_env_bool("DEBUG_UPSTREAM_STREAM", False),
        debug_upstream_max_chars=int(os.environ.get("DEBUG_UPSTREAM_MAX_CHARS", "500")),
        defaults=GatewayDefaults(
            provider=os.environ.get("DEFAULT_PROVIDER", "google"),
            locale=os.environ.get("DEFAULT_LOCALE", "en-CN"),
            source=os.environ.get("DEFAULT_SOURCE", "quick_ai"),
            system_instruction=os.environ.get("DEFAULT_SYSTEM_INSTRUCTION", "markdown"),
            debug=os.environ.get("DEFAULT_DEBUG", "false").lower() == "true",
        ),
        raycast_signature=RaycastSignatureConfig(
            enabled=_env_bool("RAYCAST_SIGNATURE_ENABLED", False),
            signing_secret=os.environ.get("RAYCAST_SIGNING_SECRET") or None,
            secret_is_transformed=_env_bool("RAYCAST_SIGNING_SECRET_TRANSFORMED", True),
            device_id=os.environ.get("RAYCAST_DEVICE_ID") or None,
            anonymous_id=os.environ.get("RAYCAST_ANONYMOUS_ID") or None,
            jwt_expires_in=float(os.environ.get("RAYCAST_JWT_EXPIRES_IN", "60")),
            user_agent=os.environ.get("RAYCAST_USER_AGENT") or None,
            accept_language=os.environ.get("RAYCAST_ACCEPT_LANGUAGE") or None,
            experimental=os.environ.get("RAYCAST_EXPERIMENTAL") or None,
            priority=os.environ.get("RAYCAST_PRIORITY") or None,
        ),
    )


def load_env_file() -> None:
    env_file = os.environ.get("RAYCAST_GATEWAY_ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=False)
        return

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
