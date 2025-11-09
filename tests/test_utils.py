from __future__ import annotations

from ollama_proxy import utils


def test_extract_text_variants():
    assert utils.extract_text("hello") == "hello"
    assert utils.extract_text({"text": "hi"}) == "hi"
    assert utils.extract_text([{"content": "a"}, "b"]) == "a\nb"
    assert utils.extract_text(None) == ""
    assert utils.extract_text(123) == "123"


def test_responses_to_chat_body_and_generate_payload():
    payload = {
        "model": "foo",
        "input": [{"role": "user", "content": [{"text": "hello"}]}],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_output_tokens": 100,
    }
    body = utils.responses_to_chat_body(payload, base_model="base")
    assert body["model"] == "base"
    gen_payload = utils.build_generate_payload(body)
    assert gen_payload["options"]["temperature"] == 0.2


def test_title_detection():
    assert utils.detect_title_generation([{"content": "make a title under 32 characters"}]) is True
    assert utils.detect_title_generation([{"content": "normal request"}]) is False
    assert utils.detect_title_generation(None) is False


def test_anthropic_conversion():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "source": "x"},
            ],
        }
    ]
    converted = utils.anthropic_messages_to_openai(messages, system="sys")
    assert converted[0]["role"] == "system"
    assert "[Image content not supported]" in converted[1]["content"]


def test_openai_json_to_text_and_usage_helpers():
    obj = {"response": "fallback"}
    assert utils.openai_json_to_text(obj) == "fallback"
    usage = utils.openai_usage_to_responses_usage({"usage": {"input_tokens": 1}})
    assert usage["total_tokens"] == 1


def test_final_responses_object_strips_quotes():
    resp = utils.final_responses_object("model", '"quoted"', {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    assert resp["output_text"][0] == "quoted"
