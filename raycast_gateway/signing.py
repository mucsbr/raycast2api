from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any


DEVICE_TAG_SALT = "hpaJuPMz6_cRCRRo*g9BuqPDE_qt"


@dataclass(frozen=True)
class RaycastSignatureConfig:
    enabled: bool = False
    signing_secret: str | None = None
    secret_is_transformed: bool = True
    device_id: str | None = None
    anonymous_id: str | None = None
    jwt_expires_in: float = 60.0
    user_agent: str | None = None
    accept_language: str | None = None
    experimental: str | None = None
    priority: str | None = None


def serialize_json_body(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def derive_device_id(platform_uuid: str, platform_serial: str) -> str:
    material = f"{platform_uuid}{platform_serial}{DEVICE_TAG_SALT}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def transform_signature_material(value: str) -> str:
    out: list[str] = []

    for ch in value:
        if "A" <= ch <= "Z":
            out.append(chr((ord(ch) - ord("A") + 13) % 26 + ord("A")))
        elif "a" <= ch <= "z":
            out.append(chr((ord(ch) - ord("a") + 13) % 26 + ord("a")))
        elif "0" <= ch <= "9":
            out.append(chr((ord(ch) - ord("0") + 5) % 10 + ord("0")))
        else:
            out.append(ch)

    return "".join(out)


def derive_signing_secret(signing_secret: str, secret_is_transformed: bool) -> str:
    if secret_is_transformed:
        return signing_secret
    return transform_signature_material(signing_secret)


def build_raycast_signature_headers(
    serialized_body: str,
    config: RaycastSignatureConfig,
    *,
    timestamp: int | None = None,
    jwt_issued_at: float | None = None,
) -> dict[str, str]:
    if not config.enabled:
        return {}
    if not config.signing_secret:
        raise ValueError("RAYCAST_SIGNING_SECRET is required when Raycast signing is enabled")
    if not config.device_id:
        raise ValueError("RAYCAST_DEVICE_ID is required when Raycast signing is enabled")

    issued_timestamp = int(time.time()) if timestamp is None else timestamp
    timestamp_value = str(issued_timestamp)
    body_hash = hashlib.sha256(serialized_body.encode("utf-8")).hexdigest()
    raw_signing_input = f"{timestamp_value}.{config.device_id}.{body_hash}"
    transformed_secret = derive_signing_secret(
        config.signing_secret,
        config.secret_is_transformed,
    )
    transformed_signing_input = transform_signature_material(raw_signing_input)
    signature = hmac.new(
        transformed_secret.encode("utf-8"),
        transformed_signing_input.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "X-Raycast-Timestamp": timestamp_value,
        "X-Raycast-DeviceId": config.device_id,
        "X-Raycast-Signature-v2": signature,
    }

    if config.anonymous_id:
        legacy_iat = float(issued_timestamp) if jwt_issued_at is None else jwt_issued_at
        headers["X-Raycast-Signature"] = build_legacy_signature(
            transformed_secret,
            config.anonymous_id,
            legacy_iat,
            config.jwt_expires_in,
        )

    return headers


def build_legacy_signature(
    signing_secret: str,
    anonymous_id: str,
    issued_at: float,
    expires_in: float,
) -> str:
    header = {"typ": "JWT", "alg": "HS256"}
    payload = {"iat": issued_at, "exp": issued_at + expires_in, "aid": anonymous_id}
    signing_input = f"{_base64url_json(header)}.{_base64url_json(payload)}"
    mac = hmac.new(
        signing_secret.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    signature = base64.urlsafe_b64encode(mac).rstrip(b"=").decode("ascii")
    return f"{signing_input}.{signature}"


def _base64url_json(data: dict[str, object]) -> str:
    raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
