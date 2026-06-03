# Raycast Gateway

OpenAI-like `/v1/chat/completions` gateway for clients such as Raycast. The gateway accepts OpenAI-compatible chat requests, converts them to the company API request shape, then converts company API responses back to OpenAI-like responses.

## Environment

Create the conda environment first:

```bash
conda env create -f environment.yml
conda activate raycast-gateway
```

Copy the configuration template and fill in your local values:

```bash
cp .env.example .env
```

The gateway loads `.env` automatically from the project root. You can also point it at another file:

```bash
export RAYCAST_GATEWAY_ENV_FILE=/absolute/path/to/raycast-gateway.env
```

The signing fields from the Raycast signature script live in `.env`: `RAYCAST_SIGNING_SECRET`, `RAYCAST_DEVICE_ID`, `RAYCAST_ANONYMOUS_ID`, `RAYCAST_EXPERIMENTAL`, `RAYCAST_USER_AGENT`, and `RAYCAST_ACCEPT_LANGUAGE`. The gateway signs the exact compact, sorted JSON body that it sends upstream. For `GET /v1/models`, Raycast expects the v2 signature to be computed over `{}`.

To print a minimal request summary, set `LOG_REQUEST_SUMMARY=true` in `.env`. The gateway prints only provider, model, and reasoning effort. To inspect request bodies, set `DEBUG_REQUEST_BODY=true`; the gateway prints the incoming client JSON and converted company JSON with common sensitive keys redacted. Use `DEBUG_REQUEST_BODY_MAX_CHARS` to control truncation. To inspect raw upstream stream lines from Raycast, set `DEBUG_UPSTREAM_STREAM=true`. The gateway prints upstream response status, content type, and each raw SSE line to stderr. Use `DEBUG_UPSTREAM_MAX_CHARS` to control truncation.

Run the gateway:

```bash
uvicorn raycast_gateway.main:app --host 127.0.0.1 --port 8000 --reload
```

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

The container reads `.env` through `docker-compose.yml` and exposes the gateway on `http://127.0.0.1:8000`. The Docker image uses the Tsinghua Debian and PyPI mirrors for dependency installation.

## OpenAI-like Behavior

OpenAI-like tools are sent upstream as `local_tool` function definitions. The gateway does not emit `remote_tool` in company API requests; legacy `remote_tool` or `{ "name": "..." }` inputs are normalized to `local_tool`.

`GET /v1/models` proxies Raycast's model catalog and returns an OpenAI-like model list:

```json
{"object":"list","data":[{"id":"gemini-3.5-flash","object":"model","created":0,"owned_by":"google"}]}
```

Streaming responses are emitted as SSE events where every JSON payload has a full `chat.completion.chunk` envelope:

```text
data: {"id":"chatcmpl_xxx","object":"chat.completion.chunk","created":1780469337,"model":"model","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}

data: [DONE]
```

Internal `reasoning` text is mapped to the common OpenAI-like extension field:

```json
{"choices":[{"delta":{"reasoning_content":"thinking..."}}]}
```

This is intentionally an OpenAI-like extension, while the outer response shape remains OpenAI-compatible.

## Tests

```bash
pytest
```
