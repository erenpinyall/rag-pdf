import json
from typing import Generator
from config import (
    LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
)


def _ollama_stream(messages: list[dict], temperature: float = 0.3) -> Generator[str, None, None]:
    import ollama
    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=True,
        options={"temperature": temperature},
        think=False,
    )
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]


def _ollama_generate(messages: list[dict], temperature: float = 0.3) -> str:
    import ollama
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=False,
        options={"temperature": temperature},
        think=False,
    )
    return response["message"]["content"]


def _openai_stream(messages: list[dict], temperature: float = 0.3) -> Generator[str, None, None]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=2000,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _openai_generate(messages: list[dict], temperature: float = 0.3) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def llm_stream(messages: list[dict], temperature: float = 0.3) -> Generator[str, None, None]:
    provider = LLM_PROVIDER.lower()
    if provider == "openai":
        yield from _openai_stream(messages, temperature)
    else:
        yield from _ollama_stream(messages, temperature)


def llm_generate(messages: list[dict], temperature: float = 0.3) -> str:
    provider = LLM_PROVIDER.lower()
    if provider == "openai":
        return _openai_generate(messages, temperature)
    return _ollama_generate(messages, temperature)


def check_connection() -> dict:
    provider = LLM_PROVIDER.lower()
    try:
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            client.models.list()
            return {"status": "ok", "provider": "openai", "model": OPENAI_MODEL}
        else:
            import ollama
            models = ollama.list()
            model_names = [m.model for m in models.models]
            available = OLLAMA_MODEL in model_names
            return {
                "status": "ok" if available else "model_missing",
                "provider": "ollama",
                "model": OLLAMA_MODEL,
                "available_models": model_names,
            }
    except Exception as e:
        return {"status": "error", "provider": provider, "error": str(e)}
