import os
import time
import json as _json
from uuid import uuid4
from typing import Any, Dict, AsyncGenerator

import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Config ---
UPSTREAM = os.environ.get("UPSTREAM", "http://10.147.20.68:11434")
REQUIRED_BEARER = os.environ.get("PROXY_API_KEY", "ollama-local")
DEFAULT_BASE_MODEL = os.environ.get("PROXY_BASE_MODEL", "gpt-oss:20b")
TITLE_MODEL = os.environ.get("PROXY_TITLE_MODEL", "granite3.1-moe:1b")
TIMEOUT = httpx.Timeout(300.0, connect=10.0, read=300.0)  # 5 minutes for DeepSeek R1 reasoning

# --- App ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------
def extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                parts.append(c.get("text") or c.get("content") or "")
        return "\n".join(p for p in parts if p)
    return str(content)

def responses_to_chat_body(src: Dict[str, Any]) -> Dict[str, Any]:
    input_arr = src.get("input") or []
    messages = []
    for item in input_arr:
        role = item.get("role", "user")
        content = extract_text(item.get("content"))
        messages.append({"role": role, "content": content})
    
    # Store original requested model for response
    original_model = src.get("model", "gpt-4o")
    
    # Always use configured base model as the actual model (will be overridden for titles)
    actual_model = DEFAULT_BASE_MODEL
    
    body = {
        "model": actual_model,
        "original_model": original_model,  # Store for response
        "messages": messages,
        "temperature": src.get("temperature"),
        "top_p": src.get("top_p"),
        "max_tokens": src.get("max_output_tokens"),
        "stream": bool(src.get("stream", True)),
    }
    # prune None
    return {k: v for k, v in body.items() if v is not None}

def messages_to_prompt(messages):
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)

def build_generate_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    prompt = messages_to_prompt(body.get("messages", []))
    gen = {
        "model": body["model"],
        "prompt": prompt,
        "stream": body.get("stream", False),
        "options": {}
    }
    if "temperature" in body and body["temperature"] is not None:
        gen["options"]["temperature"] = body["temperature"]
    if "top_p" in body and body["top_p"] is not None:
        gen["options"]["top_p"] = body["top_p"]
    if "max_tokens" in body and body["max_tokens"] is not None:
        gen["options"]["num_predict"] = body["max_tokens"]
    return gen

def sse_event(event_type: str, data_obj: dict) -> bytes:
    # Responses API SSE: "event: <type>\n" + "data: <json>\n\n"
    return f"event: {event_type}\n" + "data: " + _json.dumps(data_obj, ensure_ascii=False) + "\n\n"

def openai_json_to_text(obj: Dict[str, Any]) -> str:
    """
    Extracts text from a non-stream OpenAI chat completion object.
    """
    try:
        if "choices" in obj and obj["choices"]:
            ch0 = obj["choices"][0]
            if "message" in ch0 and isinstance(ch0["message"], dict):
                return ch0["message"].get("content") or ""
            if "delta" in ch0 and isinstance(ch0["delta"], dict):
                return ch0["delta"].get("content") or ""
    except Exception:
        pass
    # Ollama /api/generate non-stream shape: {"response": "...", ...}
    if "response" in obj and isinstance(obj["response"], str):
        return obj["response"]
    return ""

def openai_usage_to_responses_usage(obj: Dict[str, Any]) -> Dict[str, int]:
    usage = obj.get("usage") or {}
    # Map best-effort; Responses API typically expects input_tokens/output_tokens/total_tokens
    it = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    ot = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    tt = usage.get("total_tokens") or (it + ot)
    return {"input_tokens": it, "output_tokens": ot, "total_tokens": tt}

def final_responses_object(model: str, text: str, usage: Dict[str, int]) -> Dict[str, Any]:
    """
    Create a proper Responses API object for non-streaming responses.
    Based on OpenAI Responses API format - must match working proxy.py format exactly.
    """
    resp_id = f"resp_{uuid4().hex}"
    item_id = f"msg_{uuid4().hex}"
    
    # Clean up text: remove surrounding quotes if present
    cleaned_text = text.strip()
    if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
        cleaned_text = cleaned_text[1:-1]
    if cleaned_text.startswith("'") and cleaned_text.endswith("'"):
        cleaned_text = cleaned_text[1:-1]
    
    created_ts = int(time.time())
    return {
        "id": resp_id,
        "object": "response",
        "created": created_ts,
        "created_at": created_ts,
        "model": model,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "usage": usage,
        "metadata": {},
        "output_text": [cleaned_text],
        "output": [
            {
                "id": item_id,
                "object": "response.output_item",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": cleaned_text
                    }
                ]
            }
        ]
    }

# ---------- Streaming path: translate to Responses API SSE ----------
async def stream_chat_or_fallback_responses_api(body: Dict[str, Any]):
    """
    Stream upstream (OpenAI chat or Ollama /api/generate) -> Responses API SSE
    Uses simplified SSE format like working proxy.py
    """
    response_id = f"resp_{uuid4().hex}"
    item_id = f"item_{uuid4().hex}"
    model = body.get("model")
    created_ts = int(time.time())
    output_index = 0
    content_index = 0
    full_content = ""

    # Simple data: format (no event: lines)
    def ev(payload: dict) -> str:
        return f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n"

    # 1) response.created
    yield ev({
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created": created_ts,
            "status": "in_progress"
        }
    })

    # 2) response.output_item.added
    yield ev({
        "type": "response.output_item.added",
        "response_id": response_id,
        "output_index": output_index,
        "item": {
            "id": item_id,
            "object": "response.output_item",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": []
        }
    })

    # 3) Stream deltas - try OpenAI chat first, fallback to Ollama generate
    first_delta = True
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream("POST", f"{UPSTREAM}/v1/chat/completions", json=dict(body, stream=True)) as r:
                # Handle non-stream JSON response
                ctype = (r.headers.get("Content-Type") or "").lower()
                if "application/json" in ctype:
                    data = await r.aread()
                    try:
                        obj = _json.loads(data.decode("utf-8", "ignore"))
                        ch0 = (obj.get("choices") or [{}])[0]
                        text = ""
                        if isinstance(ch0.get("message"), dict):
                            text = ch0["message"].get("content") or ""
                        elif isinstance(ch0.get("delta"), dict):
                            text = ch0["delta"].get("content") or ""
                        if text:
                            # Emit content_part.added before first delta
                            yield ev({
                                "type": "response.content_part.added",
                                "response_id": response_id,
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": content_index,
                                "part": {"type": "output_text", "text": ""}
                            })
                            yield ev({
                                "type": "response.output_text.delta",
                                "response_id": response_id,
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": content_index,
                                "delta": text
                            })
                            full_content = text
                            first_delta = False
                    except Exception:
                        pass
                else:
                    # Stream SSE response
                    r.raise_for_status()
                    async for raw in r.aiter_lines():
                        if not raw:
                            continue
                        line = raw[6:] if raw.startswith("data: ") else raw
                        if line.strip() == "[DONE]":
                            break
                        try:
                            obj = _json.loads(line)
                        except Exception:
                            continue
                        ch0 = (obj.get("choices") or [{}])[0]
                        delta = ch0.get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            if first_delta:
                                # Emit content_part.added before first delta
                                yield ev({
                                    "type": "response.content_part.added",
                                    "response_id": response_id,
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "content_index": content_index,
                                    "part": {"type": "output_text", "text": ""}
                                })
                                first_delta = False
                            yield ev({
                                "type": "response.output_text.delta",
                                "response_id": response_id,
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": content_index,
                                "delta": piece
                            })
                            full_content += piece
                        if ch0.get("finish_reason"):
                            break
    except httpx.HTTPStatusError as e:
        if not (e.response and e.response.status_code == 404):
            raise
        # Fallback: Ollama /api/generate
        gen = build_generate_payload(dict(body, stream=True))
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream("POST", f"{UPSTREAM}/api/generate", json=gen) as r:
                r.raise_for_status()
                async for raw in r.aiter_lines():
                    if not raw:
                        continue
                    try:
                        obj = _json.loads(raw)
                    except Exception:
                        continue
                    piece = obj.get("response")
                    if piece:
                        if first_delta:
                            # Emit content_part.added before first delta
                            yield ev({
                                "type": "response.content_part.added",
                                "response_id": response_id,
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": content_index,
                                "part": {"type": "output_text", "text": ""}
                            })
                            first_delta = False
                        yield ev({
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": content_index,
                            "delta": piece
                        })
                        full_content += piece

    # 4) response.content_part.done
    yield ev({
        "type": "response.content_part.done",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": {"type": "output_text", "text": full_content}
    })

    # 5) response.output_item.done
    yield ev({
        "type": "response.output_item.done",
        "response_id": response_id,
        "output_index": output_index,
        "item": {
            "id": item_id,
            "object": "response.output_item",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": full_content}]
        }
    })

    # 6) response.completed (with usage info)
    yield ev({
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "created": created_ts,
            "created_at": created_ts,
            "status": "completed",
            "output_text": [full_content],
            "output": [{
                "id": item_id,
                "object": "response.output_item",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": full_content}]
            }],
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
        }
    })

# ---------- Non-stream path: return a single Responses API JSON ----------
async def call_chat_or_fallback_responses_final(body: Dict[str, Any], truncate_title: bool = False) -> Dict[str, Any]:
    """
    Non-streaming call:
      - Try /v1/chat/completions (stream=false)
      - On 404, fallback to /api/generate (stream=false)
      - Convert result into a single Responses API object
    Returns: Dict that FastAPI will automatically convert to JSON
    """
    # Ensure non-stream upstream
    body_local = dict(body)
    body_local["stream"] = False

    # Try OpenAI-compatible
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(f"{UPSTREAM}/v1/chat/completions", json=body_local)
            print(f"[DEBUG] call_chat_or_fallback: Upstream status {r.status_code}")
            print(f"[DEBUG] call_chat_or_fallback: Response headers {dict(r.headers)}")
            
            r.raise_for_status()
            obj = r.json()
            
            print(f"[DEBUG] call_chat_or_fallback: Raw upstream response:")
            print(f"[DEBUG] {_json.dumps(obj, indent=2)}")
            
            text = openai_json_to_text(obj)
            
            # Apply title truncation if requested
            if truncate_title and len(text) > 32:
                text = text[:32].strip()
                print(f"[INFO] Truncated title to 32 chars: '{text}'")
            
            print(f"[DEBUG] call_chat_or_fallback: Extracted text (first 200 chars): {text[:200]}")
            
            usage = openai_usage_to_responses_usage(obj)
            # Use original requested model in response
            response_model = body_local.get("original_model", body_local.get("model"))
            result = final_responses_object(response_model, text, usage)
            print(f"[DEBUG] Created responses object from upstream chat/completions: {_json.dumps(result, indent=2)}")
            print(f"[DEBUG] Returning dict directly (FastAPI will auto-convert to JSON)")
            return result
    except httpx.HTTPStatusError as e:
        if e.response is None or e.response.status_code != 404:
            print(f"[ERROR] Upstream error (not 404): status={e.response.status_code if e.response else 'none'}")
            print(f"[ERROR] Response body: {e.response.text if e.response else 'no body'}")
            raise

    # Fallback to /api/generate
    print(f"[DEBUG] Falling back to /api/generate for non-streaming responses")
    gen = build_generate_payload(body_local)
    print(f"[DEBUG] Generate payload: {_json.dumps(gen, indent=2)}")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client2:
        r2 = await client2.post(f"{UPSTREAM}/api/generate", json=gen)
        r2.raise_for_status()
        obj2 = r2.json()
        
        print(f"[DEBUG] Ollama /api/generate raw response:")
        print(f"[DEBUG] {_json.dumps(obj2, indent=2)}")
        
        
        text2 = "Test Response back"
        
        # Apply title truncation if requested
        if truncate_title and len(text2) > 32:
            text2 = text2[:32].strip()
            print(f"[INFO] Truncated title to 32 chars: '{text2}'")
        
        print(f"[DEBUG] Extracted text from Ollama (first 200 chars): {text2[:200]}")
        
        # Ollama generate rarely reports token usage; send zeros if missing
        usage2 = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        # Use original requested model in response
        response_model2 = body_local.get("original_model", body_local.get("model"))
        result2 = final_responses_object(response_model2, text2, usage2)
        print(f"[DEBUG] Created responses object from Ollama generate: {_json.dumps(result2, indent=2)}")
        print(f"[DEBUG] Returning dict directly (FastAPI will auto-convert to JSON)")
        return result2

# ---------- Routes ----------
@app.post("/v1/responses")
async def responses(request: Request, authorization: str = Header(default="")):
    # 1) API key enforcement
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.split(" ", 1)[1] != REQUIRED_BEARER:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 2) Parse incoming Responses API request
    try:
        src = await request.json()
        print(f"[DEBUG] /v1/responses received: model={src.get('model')}, stream={src.get('stream')}, input={len(src.get('input', []))} msgs")
        print(f"[DEBUG] Full incoming request body from AFFiNE:")
        print(f"[DEBUG] {_json.dumps(src, indent=2)}")
    except Exception as e:
        print(f"[ERROR] Failed to parse /v1/responses request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    body = responses_to_chat_body(src)
    
    # Check if this is a title generation request and override model if needed
    # Always use lightweight model for title generation
    is_title_generation = False
    if src.get("input"):
        for msg in src.get("input", []):
            content = str(msg.get("content", ""))
            if "title" in content.lower() or "summarize" in content.lower():
                if ("16 words" in content or "32 characters" in content):
                    is_title_generation = True
                    break
    
    if is_title_generation:
        actual_model = body.get("model")
        # Always override with a fast model for titles
        body["model"] = TITLE_MODEL
        print(f"[INFO] Title generation detected! Overriding model {actual_model} -> {body['model']}")
    else:
        print(
            f"[INFO] Regular chat request - using {DEFAULT_BASE_MODEL}, will return as '{body.get('original_model', 'gpt-4o')}'"
        )
    
    # Handle stream parameter: None or False should be non-streaming
    stream_param = src.get("stream")
    if stream_param is None or stream_param is False:
        stream = False
    else:
        stream = bool(stream_param)
    
    print(f"[DEBUG] Determined stream mode: {stream} (from stream_param={stream_param})")

    # 3) Dispatch
    if stream:
        return StreamingResponse(
            stream_chat_or_fallback_responses_api(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        print(f"[DEBUG] Non-streaming /v1/responses request")
        
        if is_title_generation:
            print("[TITLE] Generating chat title via Responses API envelope")

        # Always return Responses API shape so downstream Vercel SDK can parse it
        result = await call_chat_or_fallback_responses_final(body, truncate_title=is_title_generation)

        print(f"[DEBUG] Returning responses object: {_json.dumps(result)[:500]}...")
        print(f"[DEBUG] Result type: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")

        # Try serializing to verify it's valid JSON
        try:
            test_json = _json.dumps(result)
            print(f"[DEBUG] JSON serialization test: SUCCESS (length={len(test_json)})")
        except Exception as e:
            print(f"[ERROR] JSON serialization test FAILED: {e}")

        return JSONResponse(content=result)

@app.get("/health")
async def health():
    return {"ok": True, "upstream": UPSTREAM}

@app.get("/v1/models")
async def list_models(authorization: str = Header(default="")):
    """
    List available models endpoint.
    Returns a list of available models from Ollama.
    """
    # API key enforcement (optional for models list, but let's be consistent)
    if authorization and authorization.startswith("Bearer "):
        bearer_token = authorization.split(" ", 1)[1]
        if bearer_token != REQUIRED_BEARER:
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(f"{UPSTREAM}/api/tags")
            r.raise_for_status()
            ollama_models = r.json()
            
            # Convert Ollama format to OpenAI format
            models = []
            for model in ollama_models.get("models", []):
                models.append({
                    "id": model.get("name"),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama"
                })
            
            return {
                "object": "list",
                "data": models
            }
    except Exception as e:
        print(f"[ERROR] Failed to list models: {e}")
        # Return a default model if we can't fetch from Ollama
        return {
            "object": "list",
            "data": [{
                "id": "gpt-4o",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai"
            }]
        }

@app.post("/v1/embeddings")
async def embeddings(request: Request, authorization: str = Header(default="")):
    """
    OpenAI-compatible embeddings endpoint.
    Forwards to Ollama's /api/embeddings endpoint.
    """
    # 1) API key enforcement
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.split(" ", 1)[1] != REQUIRED_BEARER:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 2) Parse request
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    inputs = payload.get("input", [])
    model = payload.get("model", "gpt-4o")
    
    if isinstance(inputs, str):
        inputs = [inputs]
    if not inputs:
        raise HTTPException(400, "Missing input")

    # 3) Try Ollama embeddings endpoint first
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Ollama embeddings API format
            ollama_payload = {
                "model": model,
                "prompt": inputs[0] if len(inputs) == 1 else inputs
            }
            
            r = await client.post(
                f"{UPSTREAM}/api/embeddings",
                json=ollama_payload
            )
            r.raise_for_status()
            data = r.json()
            
            # Convert Ollama format to OpenAI format
            embedding = data.get("embedding", [])
            
            return {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": embedding if i == 0 else []
                    }
                    for i in range(len(inputs))
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0
                }
            }
    except httpx.HTTPStatusError as e:
        # If Ollama doesn't support embeddings, return a mock response
        if e.response and e.response.status_code == 404:
            print(f"[WARN] Ollama embeddings not available, returning mock embeddings")
            return {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": [0.0] * 1536  # Mock 1536-dim embedding
                    }
                    for i in range(len(inputs))
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0
                }
            }
        raise HTTPException(500, f"Embedding error: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Embeddings error: {str(e)}")
        raise HTTPException(500, f"Embedding error: {str(e)}")

@app.post("/v1/messages")
async def anthropic_messages(request: Request, authorization: str = Header(default=""), x_api_key: str = Header(default="")):
    """
    Anthropic Messages API endpoint.
    Converts Anthropic format to Ollama and returns Anthropic-compatible response.
    """
    # 1) API key enforcement - check both Bearer and x-api-key headers
    api_key = None
    if authorization.startswith("Bearer "):
        api_key = authorization.split(" ", 1)[1]
    elif x_api_key:
        api_key = x_api_key
    
    if not api_key or api_key != REQUIRED_BEARER:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 2) Parse incoming Anthropic Messages request
    try:
        payload = await request.json()
        print(f"[DEBUG] /v1/messages received: model={payload.get('model')}, messages={len(payload.get('messages', []))} msgs, stream={payload.get('stream', False)}")
    except Exception as e:
        print(f"[ERROR] Failed to parse /v1/messages request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    model = payload.get("model", "gpt-4o")
    messages = payload.get("messages", [])
    system = payload.get("system", "")
    max_tokens = payload.get("max_tokens", 4096)
    temperature = payload.get("temperature")
    stream = payload.get("stream", False)
    
    # Store original model, but always use configured base model as actual model
    original_model = model
    actual_model = DEFAULT_BASE_MODEL
    print(f"[INFO] /v1/messages: Requested '{original_model}', using '{DEFAULT_BASE_MODEL}'")

    # Convert Anthropic messages to OpenAI format
    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        
        # Handle content that might be a list or string
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        text_parts.append("[Image content not supported]")
                elif isinstance(item, str):
                    text_parts.append(item)
            text_content = "\n".join(text_parts)
        else:
            text_content = str(content) if content else ""
        
        openai_messages.append({"role": role, "content": text_content})

    # Build upstream request
    upstream_body = {
        "model": actual_model,
        "messages": openai_messages,
        "stream": stream
    }
    if temperature is not None:
        upstream_body["temperature"] = temperature
    if max_tokens:
        upstream_body["max_tokens"] = max_tokens

    # 3) Call upstream and convert response
    if stream:
        # Streaming response
        async def stream_anthropic():
            message_id = f"msg_{uuid4().hex}"
            full_content = ""
            
            try:
                async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                    async with client.stream("POST", f"{UPSTREAM}/v1/chat/completions", json=upstream_body) as r:
                        r.raise_for_status()
                        
                        # Send initial event with original requested model
                        yield f"event: message_start\ndata: {_json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': original_model, 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                        
                        # Send content_block_start
                        yield f"event: content_block_start\ndata: {_json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        
                        async for raw in r.aiter_lines():
                            if not raw:
                                continue
                            line = raw[6:] if raw.startswith("data: ") else raw
                            if line.strip() == "[DONE]":
                                break
                            try:
                                obj = _json.loads(line)
                            except Exception:
                                continue
                            
                            ch0 = (obj.get("choices") or [{}])[0]
                            delta = ch0.get("delta") or {}
                            piece = delta.get("content")
                            
                            if piece:
                                full_content += piece
                                # Send content_block_delta
                                yield f"event: content_block_delta\ndata: {_json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': piece}})}\n\n"
                            
                            if ch0.get("finish_reason"):
                                break
                        
                        # Send content_block_stop
                        yield f"event: content_block_stop\ndata: {_json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Send message_delta with usage
                        yield f"event: message_delta\ndata: {_json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                        
                        # Send message_stop
                        yield f"event: message_stop\ndata: {_json.dumps({'type': 'message_stop'})}\n\n"
                        
            except httpx.HTTPStatusError as e:
                if e.response and e.response.status_code == 404:
                    # Fallback to Ollama generate
                    print(f"[DEBUG] Falling back to /api/generate for streaming Anthropic messages")
                    gen = build_generate_payload(upstream_body)
                    gen["stream"] = True
                    
                    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                        async with client.stream("POST", f"{UPSTREAM}/api/generate", json=gen) as r:
                            r.raise_for_status()
                            
                            # Send initial events with original requested model
                            yield f"event: message_start\ndata: {_json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': original_model, 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                            yield f"event: content_block_start\ndata: {_json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                            
                            async for raw in r.aiter_lines():
                                if not raw:
                                    continue
                                try:
                                    obj = _json.loads(raw)
                                except Exception:
                                    continue
                                
                                piece = obj.get("response")
                                if piece:
                                    full_content += piece
                                    yield f"event: content_block_delta\ndata: {_json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': piece}})}\n\n"
                            
                            # Send completion events
                            yield f"event: content_block_stop\ndata: {_json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            yield f"event: message_delta\ndata: {_json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                            yield f"event: message_stop\ndata: {_json.dumps({'type': 'message_stop'})}\n\n"
                else:
                    raise
        
        return StreamingResponse(
            stream_anthropic(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Non-streaming response
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.post(f"{UPSTREAM}/v1/chat/completions", json=upstream_body)
                r.raise_for_status()
                obj = r.json()
                
                # Extract content from OpenAI format
                content = ""
                if "choices" in obj and obj["choices"]:
                    ch0 = obj["choices"][0]
                    if "message" in ch0:
                        content = ch0["message"].get("content", "")
                
                # Convert to Anthropic format with original requested model
                anthropic_response = {
                    "id": f"msg_{uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        }
                    ],
                    "model": original_model,
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": obj.get("usage", {}).get("prompt_tokens", 0),
                        "output_tokens": obj.get("usage", {}).get("completion_tokens", 0)
                    }
                }
                
                print(f"[DEBUG] Returning Anthropic messages response: {_json.dumps(anthropic_response)[:500]}...")
                return JSONResponse(content=anthropic_response)
                
        except httpx.HTTPStatusError as e:
            if e.response and e.response.status_code == 404:
                # Fallback to Ollama generate
                print(f"[DEBUG] Falling back to /api/generate for non-streaming Anthropic messages")
                gen = build_generate_payload(upstream_body)
                gen["stream"] = False
                
                async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                    r = await client.post(f"{UPSTREAM}/api/generate", json=gen)
                    r.raise_for_status()
                    ollama_response = r.json()
                    
                    content = ollama_response.get("response", "")
                    
                    # Convert to Anthropic format with original requested model
                    anthropic_response = {
                        "id": f"msg_{uuid4().hex}",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ],
                        "model": original_model,
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": ollama_response.get("prompt_eval_count", 0),
                            "output_tokens": ollama_response.get("eval_count", 0)
                        }
                    }
                    
                    print(f"[DEBUG] Returning Anthropic messages response from Ollama: {_json.dumps(anthropic_response)[:500]}...")
                    return JSONResponse(content=anthropic_response)
            else:
                print(f"[ERROR] Upstream error: {e}")
                raise HTTPException(500, f"Upstream error: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str = Header(default="")):
    """
    OpenAI-compatible chat completions endpoint.
    Returns standard OpenAI format (NOT Responses API format).
    """
    # 1) API key enforcement
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.split(" ", 1)[1] != REQUIRED_BEARER:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 2) Parse incoming chat completions request
    try:
        body = await request.body()
        if not body:
            raise HTTPException(400, "Empty request body")
        payload = _json.loads(body)
        print(f"[DEBUG] /v1/chat/completions received: model={payload.get('model')}, stream={payload.get('stream')}, messages={len(payload.get('messages', []))} msgs")
    except Exception as e:
        print(f"[ERROR] Failed to parse chat/completions request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    messages = payload.get("messages", [])
    original_model = payload.get("model", "gpt-4o")
    stream = payload.get("stream", False)
    temperature = payload.get("temperature")
    max_tokens = payload.get("max_tokens")
    
    # Always use configured base model as actual model
    actual_model = DEFAULT_BASE_MODEL
    print(f"[INFO] /v1/chat/completions: Requested '{original_model}', using '{DEFAULT_BASE_MODEL}'")

    # Build request body for upstream
    upstream_body = {
        "model": actual_model,
        "messages": messages,
        "stream": stream
    }
    if temperature is not None:
        upstream_body["temperature"] = temperature
    if max_tokens is not None:
        upstream_body["max_tokens"] = max_tokens

    # 3) Forward to upstream OpenAI-compatible endpoint
    try:
        if stream:
            print(f"[DEBUG] Forwarding streaming request to {UPSTREAM}/v1/chat/completions")
            # For streaming, proxy the SSE stream directly
            async def forward_stream():
                async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                    async with client.stream(
                        "POST",
                        f"{UPSTREAM}/v1/chat/completions",
                        json=upstream_body
                    ) as r:
                        r.raise_for_status()
                        async for chunk in r.aiter_bytes():
                            yield chunk
            
            return StreamingResponse(
                forward_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            print(f"[DEBUG] Forwarding non-streaming request to {UPSTREAM}/v1/chat/completions")
            # For non-streaming, return standard OpenAI JSON format
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.post(
                    f"{UPSTREAM}/v1/chat/completions",
                    json=upstream_body
                )
                print(f"[DEBUG] Upstream status code: {r.status_code}")
                print(f"[DEBUG] Upstream headers: {dict(r.headers)}")
                
                r.raise_for_status()
                response_data = r.json()
                
                # Replace model in response with original requested model
                response_data["model"] = original_model
                
                print(f"[DEBUG] Full upstream response JSON:")
                print(f"[DEBUG] {_json.dumps(response_data, indent=2)}")
                print(f"[DEBUG] Response type: {type(response_data)}")
                print(f"[DEBUG] Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'not a dict'}")
                
                return JSONResponse(content=response_data)
                
    except httpx.HTTPStatusError as e:
        print(f"[DEBUG] Upstream returned {e.response.status_code if e.response else 'no response'}, trying fallback")
        print(f"[DEBUG] Error response body: {e.response.text if e.response else 'no body'}")
        if e.response and e.response.status_code == 404:
            # Fallback: Use /api/generate and convert to OpenAI format
            print(f"[DEBUG] Using Ollama /api/generate fallback")
            gen_payload = build_generate_payload({
                "model": actual_model,
                "messages": messages,
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.post(f"{UPSTREAM}/api/generate", json=gen_payload)
                r.raise_for_status()
                ollama_response = r.json()
                
                print(f"[DEBUG] Ollama /api/generate raw response:")
                print(f"[DEBUG] {_json.dumps(ollama_response, indent=2)}")
                
                # Convert Ollama format to OpenAI format with original requested model
                content = ollama_response.get("response", "")
                print(f"[DEBUG] Extracted content: {content[:200]}...")
                
                openai_format = {
                    "id": f"chatcmpl-{uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": original_model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                        "completion_tokens": ollama_response.get("eval_count", 0),
                        "total_tokens": ollama_response.get("prompt_eval_count", 0) + ollama_response.get("eval_count", 0)
                    }
                }
                print(f"[DEBUG] Converted to OpenAI format:")
                print(f"[DEBUG] {_json.dumps(openai_format, indent=2)}")
                return JSONResponse(content=openai_format)
        print(f"[ERROR] Upstream error {e.response.status_code if e.response else 'unknown'}: {e}")
        raise HTTPException(500, f"Upstream error: {str(e)}")
    except Exception as e:
        print(f"[ERROR] chat/completions error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Chat completions error: {str(e)}")
