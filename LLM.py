"""OpenRouter client helpers for text generation.

Usage:
  - Set environment variable `OPENROUTER_API_KEY` with your key.
  - Create `OpenRouterLLM()` and call `generate(prompt)` or
    `generate_with_context(question, chunks)` where `chunks` is a list
    of dicts with `chunk_text` and `source` keys.
"""
from __future__ import annotations

import os
from typing import List, Dict, Optional

# try to load dotenv so a local .env file is respected
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import requests
except Exception:
    requests = None


class OpenRouterLLM:
    """Light wrapper for OpenRouter chat completions.

    Reads the API key from the environment variable named by `api_key_env`
    (default: `OPENROUTER_API_KEY`).

    Usage:
        client = OpenRouterLLM()
        client.generate_with_context(question, chunks)
    """

    def __init__(self, api_key_env: str = "OPENROUTER_API_KEY", model: str = "gpt-4o-mini"):
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise EnvironmentError(f"OpenRouter API key not found in env var {api_key_env}")
        self.model = model
        self.base_url = "https://api.openrouter.ai/v1/chat/completions"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        # Build the API payload using the chat completions schema
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Ensure the `requests` dependency is present; raise a clear error if not
        if requests is None:
            raise RuntimeError("The 'requests' library is required to call OpenRouter. Install with: pip install requests")

        resp = requests.post(self.base_url, headers=self._headers(), json=payload, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")

        # Parse the JSON response and extract the assistant text
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            # some endpoints return `text` instead
            return data["choices"][0].get("text", "")

    def generate_with_context(
        self,
        question: str,
        chunks: List[Dict],
        top_k: int = 5,
        chunk_char_limit: int = 1000,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:

        # Select top_k chunks and prepare a short context for the LLM.
        # Each chunk is prefixed with a simple source header to allow citation.
        selected = chunks[:top_k]
        context_parts = []
        for c in selected:
            text = c.get("chunk_text", "")[:chunk_char_limit]
            src = c.get("source")
            header = f"Source: {src}\n" if src else ""
            context_parts.append(header + text)

        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            "Use the following context to answer the question. If the answer is not contained in the "
            "context, say you don't know rather than inventing facts.\n\nContext:\n"
            f"{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
