from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List

from uuid import uuid4


def extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    if isinstance(content, list):
        parts = [extract_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    return str(content)


def responses_to_chat_body(src: Dict[str, Any], base_model: str) -> Dict[str, Any]:
    input_arr = src.get("input") or []
    messages = []
    for item in input_arr:
        role = item.get("role", "user")
        content = extract_text(item.get("content"))
        messages.append({"role": role, "content": content})

    original_model = src.get("model", "gpt-4o")
    actual_model = base_model
    body = {
        "model": actual_model,
        "original_model": original_model,
        "messages": messages,
        "temperature": src.get("temperature"),
        "top_p": src.get("top_p"),
        "max_tokens": src.get("max_output_tokens"),
        "stream": bool(src.get("stream", True)),
    }
    return {k: v for k, v in body.items() if v is not None}


def messages_to_prompt(messages: Iterable[Dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def build_generate_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    prompt = messages_to_prompt(body.get("messages", []))
    payload: Dict[str, Any] = {
        "model": body["model"],
        "prompt": prompt,
        "stream": body.get("stream", False),
        "options": {},
    }
    if body.get("temperature") is not None:
        payload["options"]["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["options"]["top_p"] = body["top_p"]
    if body.get("max_tokens") is not None:
        payload["options"]["num_predict"] = body["max_tokens"]
    return payload


def openai_json_to_text(obj: Dict[str, Any]) -> str:
    choices = obj.get("choices") or []
    if choices:
        first = choices[0]
        if isinstance(first.get("message"), dict):
            return first["message"].get("content") or ""
        if isinstance(first.get("delta"), dict):
            return first["delta"].get("content") or ""
    if isinstance(obj.get("response"), str):
        return obj["response"]
    return ""


def openai_usage_to_responses_usage(obj: Dict[str, Any]) -> Dict[str, int]:
    usage = obj.get("usage") or {}
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total = usage.get("total_tokens") or (prompt + completion)
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": total}


def final_responses_object(model: str, text: str, usage: Dict[str, int]) -> Dict[str, Any]:
    resp_id = f"resp_{uuid4().hex}"
    item_id = f"msg_{uuid4().hex}"
    cleaned_text = text.strip().strip('"').strip("'")
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
                "content": [{"type": "output_text", "text": cleaned_text}],
            }
        ],
    }


def detect_title_generation(input_messages: List[Dict[str, Any]] | None) -> bool:
    if not input_messages:
        return False
    for message in input_messages:
        content = str(message.get("content", "")).lower()
        if "title" in content or "summarize" in content:
            if "16 words" in content or "32 characters" in content:
                return True
    return False


def anthropic_messages_to_openai(messages: List[Dict[str, Any]], system: str | None) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    if system:
        result.append({"role": "system", "content": system})
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, dict) and item.get("type") == "image":
                    parts.append("[Image content not supported]")
                elif isinstance(item, str):
                    parts.append(item)
            text = "\n".join(parts)
        else:
            text = str(content or "")
        result.append({"role": role, "content": text})
    return result
