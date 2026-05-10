#!/usr/bin/env python3
"""
Quick stream-format probe for pet_gear_note.

It reads .env and can test:
1. Anthropic-style /v1/messages stream
2. OpenAI-compatible /v1/chat/completions stream

Usage:
    python test_stream.py --mode messages
    python test_stream.py --mode chat
    python test_stream.py --mode both
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_base_url() -> str:
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        return "https://api.anthropic.com/v1"
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def get_config() -> tuple[str, str, str]:
    load_dotenv()
    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    model = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "claude-3-5-sonnet-latest"
    base_url = normalize_base_url()
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY")
    return api_key, model, base_url


def iter_sse(resp: requests.Response, max_events: int) -> tuple[list[str], list[dict]]:
    raw_events: list[str] = []
    parsed_events: list[dict] = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        print(f"RAW: {line}")
        if not line.startswith("data: "):
            continue
        data_str = line[len("data: "):].strip()
        raw_events.append(data_str)
        if data_str == "[DONE]":
            break
        try:
            parsed = json.loads(data_str)
        except json.JSONDecodeError:
            parsed = {"_unparsed": data_str}
        parsed_events.append(parsed)
        if len(raw_events) >= max_events:
            break
    return raw_events, parsed_events


def extract_text(parsed_events: list[dict]) -> str:
    chunks: list[str] = []
    for event in parsed_events:
        if event.get("type") == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                chunks.append(delta.get("text", ""))
        elif "choices" in event:
            for choice in event["choices"]:
                delta = choice.get("delta", {})
                message = choice.get("message", {})
                if delta.get("content"):
                    chunks.append(delta["content"])
                elif message.get("content"):
                    chunks.append(message["content"])
        elif event.get("content"):
            content = event["content"]
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("text"):
                        chunks.append(part["text"])
    return "".join(chunks)


def test_messages(api_key: str, model: str, base_url: str, max_events: int) -> None:
    url = f"{base_url}/messages"
    print("\n=== Testing Anthropic /v1/messages stream ===")
    print(f"URL: {url}")
    print(f"MODEL: {model}")
    resp = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # Anthropic native API may require this header; compatible proxies usually ignore it.
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": model,
            "max_tokens": 256,
            "stream": True,
            "system": "Reply briefly.",
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        },
        timeout=120,
        stream=True,
    )
    print(f"STATUS: {resp.status_code}")
    print(f"CONTENT-TYPE: {resp.headers.get('content-type')}")
    if resp.status_code != 200:
        print(resp.text[:4000])
        return
    _, parsed_events = iter_sse(resp, max_events=max_events)
    text = extract_text(parsed_events)
    print("\n--- Extracted Text ---")
    print(text or "<EMPTY>")


def test_chat(api_key: str, model: str, base_url: str, max_events: int) -> None:
    url = f"{base_url}/chat/completions"
    print("\n=== Testing OpenAI-compatible /v1/chat/completions stream ===")
    print(f"URL: {url}")
    print(f"MODEL: {model}")
    resp = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": model,
            "stream": True,
            "messages": [
                {"role": "system", "content": "Reply briefly."},
                {"role": "user", "content": "Say hello in one sentence."},
            ],
        },
        timeout=120,
        stream=True,
    )
    print(f"STATUS: {resp.status_code}")
    print(f"CONTENT-TYPE: {resp.headers.get('content-type')}")
    if resp.status_code != 200:
        print(resp.text[:4000])
        return
    _, parsed_events = iter_sse(resp, max_events=max_events)
    text = extract_text(parsed_events)
    print("\n--- Extracted Text ---")
    print(text or "<EMPTY>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe LLM stream response format")
    parser.add_argument("--mode", choices=("messages", "chat", "both"), default="both")
    parser.add_argument("--max-events", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key, model, base_url = get_config()
    if args.mode in ("messages", "both"):
        test_messages(api_key, model, base_url, args.max_events)
    if args.mode in ("chat", "both"):
        test_chat(api_key, model, base_url, args.max_events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
