# Ollama Affine Proxy

![Tests](https://github.com/hectorandac/affine-ollama-proxy/workflows/Test%20and%20Coverage/badge.svg)
![Docker Build](https://github.com/hectorandac/affine-ollama-proxy/workflows/Docker%20Build%20and%20Push/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

FastAPI service that speaks the OpenAI Responses, OpenAI Chat Completions and Anthropic Messages APIs while executing the requests against a local Ollama instance. Drop it in front of AFFiNE (or any OpenAI-compatible client) to keep using your own models without sending data to paid SaaS providers.

## Features

- Transparent translation between OpenAI/Anthropic style payloads and Ollama models.
- Title-detection logic that routes short requests to a lightweight model.
- Streaming and non-streaming support for Responses, Messages and Chat Completions.
- Embeddings and model list shims for clients that expect those endpoints.
- Dockerfile for reproducible builds plus `.env` based configuration.

## Project layout

```
.
├── .archive/            # Historical single-file proxy (kept for reference)
├── src/ollama_proxy/    # Application package
├── tests/               # Pytest suite (>90% coverage)
├── Dockerfile
├── README.md
├── LICENSE              # MIT
├── .env.example         # Template configuration
└── .env                 # Local secrets (ignored by git)
```

## Getting started

1. **Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

2. **Configuration**
   - Copy `.env.example` to `.env`.
   - Adjust the variables to point at your Ollama host and change the proxy API key.
   - `.env` already contains the previously hard-coded defaults for convenience and is ignored by git.

3. **Run locally**
   ```bash
   uvicorn ollama_proxy.app:app --reload --port 8080
   ```

4. **Call the proxy**
   - Set `Authorization: Bearer <PROXY_API_KEY>` (or `x-api-key` for Anthropic).
   - The service exposes `/v1/responses`, `/v1/chat/completions`, `/v1/messages`, `/v1/models`, `/v1/embeddings`, and `/health`.

## Tests & coverage

All critical behaviors are covered by pytest. Coverage is enforced at 90%:

```bash
pytest --cov=ollama_proxy --cov-report=term-missing
```

## Docker

```bash
docker build -t affine-ollama-proxy .
docker run --rm -p 8080:8080 --env-file .env affine-ollama-proxy
```

The `.dockerignore` file keeps local-only artifacts out of the image.

## Environment variables

| Name | Description | Default (see `.env`) |
| --- | --- | --- |
| `UPSTREAM` | Base URL of Ollama (e.g., `http://localhost:11434`) | `http://10.147.20.68:11434` |
| `PROXY_API_KEY` | Shared secret required in `Authorization` or `x-api-key` headers | `ollama-local` |
| `PROXY_BASE_MODEL` | Default model used for long-form responses | `gpt-oss:20b` |
| `PROXY_TITLE_MODEL` | Smaller model for title/summarization prompts | `granite3.1-moe:1b` |
| `PROXY_TIMEOUT` | Request timeout in seconds | `300.0` |
| `PROXY_CONNECT_TIMEOUT` | Connect timeout in seconds | `10.0` |
| `PROXY_CORS_ALLOW_ORIGINS` | Comma separated list for CORS | `*` |

## License

MIT License – see [LICENSE](LICENSE) for details.
