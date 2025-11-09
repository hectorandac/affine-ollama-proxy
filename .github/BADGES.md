# Status Badges

Add these badges to your README.md to show the status of your CI/CD pipelines:

## GitHub Actions Workflows

```markdown
![Tests](https://github.com/hectorandac/affine-ollama-proxy/workflows/Test%20and%20Coverage/badge.svg)
![Docker Build](https://github.com/hectorandac/affine-ollama-proxy/workflows/Docker%20Build%20and%20Push/badge.svg)
```

## Coverage Badge (After Setup)

```markdown
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/hectorandac/YOUR_GIST_ID/raw/ollama-proxy-coverage.json)
```

## Example README Header

```markdown
# Ollama Affine Proxy

![Tests](https://github.com/hectorandac/affine-ollama-proxy/workflows/Test%20and%20Coverage/badge.svg)
![Docker Build](https://github.com/hectorandac/affine-ollama-proxy/workflows/Docker%20Build%20and%20Push/badge.svg)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/hectorandac/YOUR_GIST_ID/raw/ollama-proxy-coverage.json)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

FastAPI service that speaks the OpenAI Responses, OpenAI Chat Completions and Anthropic Messages APIs while executing the requests against a local Ollama instance.
```
