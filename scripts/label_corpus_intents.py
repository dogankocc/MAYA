#!/usr/bin/env python3
"""Add or refresh intent labels on JSONL corpora."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from corpus_jsonl import make_messages_record, write_jsonl_corpus  # noqa: E402
from intent_labels import classify_intent, normalize_intent, profile_intent_for_prompt  # noqa: E402


def extract_user_assistant(record: dict) -> tuple[str, str, str | None]:
    system: str | None = None
    user = ""
    assistant = ""
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role == "system" and not system:
                system = content
            elif role == "user" and not user:
                user = content
            elif role == "assistant" and user and not assistant:
                assistant = content
    if not user:
        user = str(record.get("instruction") or record.get("prompt") or record.get("input") or "").strip()
    if not assistant:
        assistant = str(record.get("response") or record.get("output") or "").strip()
    return user, assistant, system


def label_jsonl_file(path: Path) -> int:
    if not path.exists():
        return 0

    description = f"Intent etiketli corpus: {path.name}"
    updated: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                description = line.lstrip("#").strip() or description
                continue
            record = json.loads(line)
            user, assistant, system = extract_user_assistant(record)
            if not user or not assistant:
                continue
            intent = record.get("intent")
            if isinstance(intent, str) and intent.strip():
                resolved = normalize_intent(intent)
            else:
                resolved = profile_intent_for_prompt(user) or classify_intent(user, assistant)
            updated.append(
                json.dumps(
                    make_messages_record(user, assistant, system, resolved),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

    if not updated:
        return 0

    write_jsonl_corpus(path, updated, description)
    return len(updated)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    total = 0
    targets = [ROOT / "data" / "chat_corpus_tr.jsonl", *sorted((ROOT / "data" / "corpus").glob("*.jsonl"))]
    for path in targets:
        count = label_jsonl_file(path)
        if count:
            print(f"{path.name}: {count} intent etiketi")
            total += count

    print(f"Toplam etiketlenen ornek: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
