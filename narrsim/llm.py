"""Minimal Ollama client with seeded generation, JSON-constrained decoding and an
on-disk response cache (so that interrupted experiments can be resumed and every
reported number can be regenerated from the cached raw outputs)."""

import hashlib
import json
import os
import threading
import time

import requests


class OllamaClient:
    def __init__(self, model, base_url="http://127.0.0.1:11434", temperature=0.7,
                 max_tokens=1000, num_ctx=8192, timeout=600, cache_path=None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._cache = {}
        self.n_calls = 0
        self.n_cache_hits = 0
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["response"]
                    except Exception:
                        pass

    def _key(self, prompt, seed, temperature, json_mode, max_tokens):
        raw = json.dumps([self.model, prompt, seed, temperature, json_mode, max_tokens])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def generate(self, prompt, seed=0, temperature=None, json_mode=True, max_tokens=None):
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        key = self._key(prompt, seed, temperature, json_mode, max_tokens)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "seed": int(seed),
                "num_predict": int(max_tokens),
                "num_ctx": self.num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"
        if self.model.startswith(("qwen3", "deepseek-r1")):
            payload["think"] = False

        last_err = None
        for attempt in range(4):
            try:
                r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
                r.raise_for_status()
                text = r.json().get("response", "")
                break
            except Exception as e:  # network hiccup or model reload
                last_err = e
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"Ollama call failed after retries: {last_err}")

        self.n_calls += 1
        with self._lock:
            self._cache[key] = text
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "model": self.model, "seed": seed,
                                        "response": text}, ensure_ascii=False) + "\n")
        return text
