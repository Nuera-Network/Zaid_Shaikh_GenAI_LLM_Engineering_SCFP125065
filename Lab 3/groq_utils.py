"""Shared Groq API helpers for Lab 3 Prompt Engineering.

Calls are deliberately kept close to the Groq tutorial's chat-completions
pattern. Importing this module never requires an API key or network access.
"""

from __future__ import annotations

import json
import os
from typing import Any


DEFAULT_MODEL = "openai/gpt-oss-20b"


def make_client() -> Any:
    """Load .env and create a Groq client without exposing its API key."""
    try:
        from dotenv import load_dotenv
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "Missing packages. Install them using: pip install -r requirements.txt"
        ) from exc

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Create a .env file from .env.example."
        )
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    client: Any = None,
    response_format: dict[str, Any] | None = None,
    max_tokens: int = 1200,
) -> str:
    """Send a system/user prompt pair and return the model's answer."""
    groq_client = client if client is not None else make_client()
    kwargs: dict[str, Any] = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    completion = groq_client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned no text. Try again.")
    return content.strip()


def strict_json_schema(name: str, properties: dict[str, Any]) -> dict[str, Any]:
    """Define Groq GPT-OSS strict structured JSON output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def parse_object(response: str, expected_fields: list[str]) -> dict[str, Any]:
    """Validate parsed JSON at the Python boundary as well as at the API."""
    data = json.loads(response)
    if not isinstance(data, dict) or set(data) != set(expected_fields):
        raise ValueError("The response does not match the requested JSON fields.")
    return data

