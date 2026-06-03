from raycast_gateway.signing import (
    RaycastSignatureConfig,
    build_raycast_signature_headers,
    derive_device_id,
    serialize_json_body,
    transform_signature_material,
)


def test_serialize_json_body_matches_foundation_style():
    assert serialize_json_body({"b": "测试", "a": 1}) == '{"a":1,"b":"测试"}'


def test_raycast_signature_headers_match_reference_script():
    headers = build_raycast_signature_headers(
        '{"a":1,"b":"测试"}',
        RaycastSignatureConfig(
            enabled=True,
            signing_secret="abcXYZ09",
            secret_is_transformed=True,
            device_id="test-device-id",
            anonymous_id="anon-test",
        ),
        timestamp=1700000000,
    )

    assert headers == {
        "X-Raycast-Timestamp": "1700000000",
        "X-Raycast-DeviceId": "test-device-id",
        "X-Raycast-Signature-v2": "040de8db3bec88c71d38f28f0174745aa8218603719c37ac7a6dbb7c24d21869",
        "X-Raycast-Signature": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE3MDAwMDAwMDAuMCwiZXhwIjoxNzAwMDAwMDYwLjAsImFpZCI6ImFub24tdGVzdCJ9.xh3l35GTII58qcB-BVgS6fP2apqRVz41eCh-lRqKC9k",
    }


def test_raycast_models_get_signature_uses_empty_object_body():
    headers = build_raycast_signature_headers(
        "{}",
        RaycastSignatureConfig(
            enabled=True,
            signing_secret="abcXYZ09",
            secret_is_transformed=True,
            device_id="test-device-id",
            anonymous_id="anon-test",
        ),
        timestamp=1780430483,
    )

    assert headers["X-Raycast-Signature-v2"] == (
        "8e76d1d6139baf5fa6729ca0fcfb31c0f3b67db178af85a0ca60d2c80f942ee1"
    )


def test_signature_can_be_disabled():
    assert build_raycast_signature_headers("{}", RaycastSignatureConfig()) == {}


def test_transform_and_device_id_helpers():
    assert transform_signature_material("abcXYZ09.-") == "nopKLM54.-"
    assert (
        derive_device_id("uuid", "serial")
        == "11fa8ae7ed3b727400f2760a0aa91d0857e0dcd29f1ac8d476ac3e1e1aacf3a9"
    )
