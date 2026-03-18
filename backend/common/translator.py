"""
translator.py
Module: Vietnamese → English translation
Provider: Gemini Flashlite 2.5 via OpenRouter
"""

import os
import requests
from typing import Optional

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class Translator:
    """
    Translation wrapper class
    Currently supports:
        - vi → en translation
    """

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash-lite",
        timeout: int = 60,
        cache: bool = False
    ):
        self.model = model
        self.timeout = timeout
        self.cache_enabled = cache
        self.cache_dict = {}

        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables"
            )

        self.url = "https://openrouter.ai/api/v1/chat/completions"

    # =========================
    # Core translate function
    # =========================
    def vi_to_en(self, text_vi: str) -> str:
        """
        Translate Vietnamese → English
        """

        if not text_vi:
            return ""

        # ---- Cache check ----
        if self.cache_enabled and text_vi in self.cache_dict:
            return self.cache_dict[text_vi]

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Translate the following Vietnamese claim to English. "
                        "Preserve factual meaning. Do not explain.\n\n"
                        f"{text_vi}"
                    ),
                }
            ],
            "temperature": 0
        }

        try:
            resp = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()

            data = resp.json()
            translated = data["choices"][0]["message"]["content"].strip()

            if self.cache_enabled:
                self.cache_dict[text_vi] = translated

            return translated

        except Exception as e:
            print(f"[Translator] Error → fallback original: {e}")
            return text_vi

    # =========================
    # Batch translate (optional)
    # =========================
    def batch_vi_to_en(self, texts):
        """
        Translate list of Vietnamese texts
        """
        return [self.vi_to_en(t) for t in texts]