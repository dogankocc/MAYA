"""Shared helpers for SFT JSONL corpus files (messages + intent format)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from intent_labels import classify_intent, normalize_intent, profile_intent_for_prompt

DEFAULT_SYSTEM_PROMPT = (
    "Sen Turkce konusan, teknik konularda net ve yardimci bir yapay zeka asistanisin."
)


def make_messages_record(
    user: str,
    assistant: str,
    system: Optional[str] = None,
    intent: Optional[str] = None,
) -> dict:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})

    resolved_intent = intent or profile_intent_for_prompt(user) or classify_intent(user, assistant)
    return {"intent": normalize_intent(resolved_intent), "messages": messages}


def format_jsonl_line(
    user: str,
    assistant: str,
    system: Optional[str] = None,
    intent: Optional[str] = None,
) -> str:
    return json.dumps(
        make_messages_record(user, assistant, system, intent),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_jsonl_corpus(path: Path, lines: Iterable[str], description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# {description}\n")
        for line in lines:
            handle.write(line + "\n")


def parse_pipe_line(line: str) -> Optional[tuple[str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "|" not in stripped:
        return None
    prompt, response = stripped.split("|", 1)
    prompt = prompt.strip()
    response = response.strip()
    if not prompt or not response:
        return None
    return prompt, response


def convert_pipe_file_to_jsonl(
    source: Path,
    destination: Path,
    system: Optional[str] = None,
) -> int:
    if not source.exists():
        return 0

    lines: list[str] = []
    description = f"Migrated from {source.name}"
    with source.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.lstrip().startswith("#"):
                description = raw.lstrip("#").strip() or description
                continue
            pair = parse_pipe_line(raw)
            if pair is None:
                continue
            user, assistant = pair
            lines.append(format_jsonl_line(user, assistant, system))

    if not lines:
        return 0

    write_jsonl_corpus(destination, lines, description)
    return len(lines)
