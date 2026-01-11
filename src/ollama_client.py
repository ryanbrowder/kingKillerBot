# src/ollama_client.py
from __future__ import annotations

from typing import Any, Dict, Optional
import requests


def ollama_chat(
    model: str,
    system: str,
    user: str,
    host: str = "http://localhost:11434",
    temperature: float = 0.2,
) -> str:
    """
    Calls Ollama's /api/chat endpoint and returns assistant message content.
    """
    url = f"{host}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]