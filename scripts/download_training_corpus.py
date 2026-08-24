#!/usr/bin/env python3
"""Download popular Turkish dialogue/reasoning datasets for local LLM training."""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from corpus_jsonl import format_jsonl_line, write_jsonl_corpus  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "data" / "corpus"

# Favor longer answers for reasoning / fluent Turkish.
PROMPT_MIN = 4
PROMPT_MAX = 180
RESPONSE_MIN_SHORT = 12
RESPONSE_MIN_LONG = 36
RESPONSE_MAX = 720
TOTAL_MAX = 820

REASONING_FLAGS = (
    "has_logical_deduction",
    "has_rule_based_reasoning",
    "has_evidence_based_reasoning",
    "has_multi_source_synthesis",
    "has_hybrid_complex_logic",
    "has_basic_math",
    "has_text_based_qa",
    "has_anti_hallucination",
)


def trim_response(response: str, max_len: int = RESPONSE_MAX) -> str:
    if len(response) <= max_len:
        return response
    cut = response[:max_len]
    best = -1
    for marker in (". ", "! ", "? ", "; ", ", "):
        index = cut.rfind(marker)
        if index >= max_len // 4 and index > best:
            best = index + len(marker) - 1
    if best >= max_len // 4:
        return cut[: best + 1].strip()
    return cut.rstrip() + "..."


def trim_prompt(prompt: str, max_len: int = PROMPT_MAX) -> str:
    if len(prompt) <= max_len:
        return prompt
    cut = prompt[:max_len]
    index = cut.rfind(" ")
    if index >= max_len // 2:
        return cut[:index].strip()
    return cut.strip()


def normalize(text: object) -> str:
    if text is None:
        return ""
    value = str(text).replace("\r", " ").replace("\n", " ").replace("|", "/")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_user_assistant(conversations: object) -> Tuple[str, str]:
    user = ""
    assistant = ""
    if not isinstance(conversations, list):
        return user, assistant
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "user" and not user:
            user = normalize(content)
        elif role == "assistant" and user and not assistant:
            assistant = normalize(content)
            break
    return user, assistant


def add_pair(
    prompt: str,
    response: str,
    out: list[str],
    seen: Set[str],
    *,
    prefer_long: bool = False,
) -> bool:
    prompt = trim_prompt(normalize(prompt))
    response = trim_response(normalize(response))
    min_response = RESPONSE_MIN_LONG if prefer_long else RESPONSE_MIN_SHORT
    if not prompt or not response:
        return False
    if len(prompt) < PROMPT_MIN or len(response) < min_response:
        return False
    if len(prompt) + len(response) + 1 > TOTAL_MAX:
        response = trim_response(response, max(TOTAL_MAX - len(prompt) - 1, min_response))
    if len(response) < min_response:
        return False
    key = prompt.casefold()
    if key in seen:
        return False
    seen.add(key)
    out.append(format_jsonl_line(prompt, response))
    return True


def write_corpus(path: Path, lines: list[str], header: str) -> None:
    write_jsonl_corpus(path, lines, header)


def collect_from_indices(
    rows: Iterable[Tuple[int, object]],
    limit: int,
    seen: Set[str],
    prompt_key: str,
    response_key: str,
    *,
    prefer_long: bool = False,
    row_filter: Optional[Callable[[object], bool]] = None,
) -> list[str]:
    lines: list[str] = []
    for _, row in rows:
        if len(lines) >= limit:
            break
        if row_filter is not None and not row_filter(row):
            continue
        prompt = row.get(prompt_key) if isinstance(row, dict) else None
        response = row.get(response_key) if isinstance(row, dict) else None
        add_pair(prompt, response, lines, seen, prefer_long=prefer_long)
    return lines


def load_sixfinger(seen: Set[str], limit: int, rng: random.Random) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("sixfingerdev/chatbot-turkish-dataset-sixfinger-2b", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    return collect_from_indices(
        ((index, ds[index]) for index in indices),
        limit,
        seen,
        "input",
        "output",
        prefer_long=True,
    )


def is_reasoning_row(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    return any(bool(row.get(flag)) for flag in REASONING_FLAGS)


def load_nafie_reasoning(seen: Set[str], limit: int, rng: random.Random) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("nafie-ai/nafie-sft-v1", split="train")
    reasoning_indices = [index for index in range(len(ds)) if is_reasoning_row(ds[index])]
    rng.shuffle(reasoning_indices)
    return collect_from_indices(
        ((index, ds[index]) for index in reasoning_indices),
        limit,
        seen,
        "prompt",
        "response",
        prefer_long=True,
        row_filter=is_reasoning_row,
    )


def load_nafie_general(seen: Set[str], limit: int, rng: random.Random) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("nafie-ai/nafie-sft-v1", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    return collect_from_indices(
        ((index, ds[index]) for index in indices),
        limit,
        seen,
        "prompt",
        "response",
        prefer_long=False,
    )


def load_ultrachat(seen: Set[str], limit: int, rng: random.Random) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("hamuz/UltraChatTR_50k", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    lines: list[str] = []
    for index in indices:
        if len(lines) >= limit:
            break
        row = ds[index]
        user, assistant = extract_user_assistant(row.get("messages"))
        if not user:
            user = normalize(row.get("instruction") or row.get("prompt"))
        if not assistant:
            assistant = normalize(row.get("output"))
        add_pair(user, assistant, lines, seen, prefer_long=True)
    return lines


def load_gemma51k(seen: Set[str], limit: int, rng: random.Random) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("afkfatih/turkish-gemma-51k", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    lines: list[str] = []
    for index in indices:
        if len(lines) >= limit:
            break
        row = ds[index]
        user, assistant = extract_user_assistant(row.get("conversations"))
        add_pair(user, assistant, lines, seen, prefer_long=True)
    return lines


def ensure_datasets() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError:
        import subprocess

        print("Installing Hugging Face datasets...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])


def download_all(out_dir: Path, seed: int) -> int:
    ensure_datasets()
    rng = random.Random(seed)
    seen: Set[str] = set()

    # Popular Turkish sets; each source has its own budget (dedup across all).
    sources: List[Tuple[str, Callable[[Set[str], int, random.Random], list[str]], int, str]] = [
        (
            "nafie_reasoning_tr.jsonl",
            load_nafie_reasoning,
            6000,
            "Nafie SFT v1 - mantik, akil yurutme, kanit temelli cevaplar",
        ),
        (
            "turkish_gemma_51k.jsonl",
            load_gemma51k,
            5000,
            "afkfatih/turkish-gemma-51k - 51K Turkce sohbet (populer)",
        ),
        (
            "ultrachat_tr.jsonl",
            load_ultrachat,
            4500,
            "UltraChatTR 50k - uzun Turkce diyaloglar",
        ),
        (
            "sixfinger_tr.jsonl",
            load_sixfinger,
            2400,
            "SixFinger 2B - temiz yuksek kalite Turkce chat",
        ),
        (
            "nafie_general_tr.jsonl",
            load_nafie_general,
            3500,
            "Nafie SFT v1 - genel talimat ve sohbet",
        ),
    ]

    total = 0
    manifest: list[str] = []
    for filename, loader, budget, description in sources:
        print(f"Downloading {filename} (hedef {budget})...", flush=True)
        print(f"  {description}", flush=True)
        try:
            lines = loader(seen, budget, rng)
        except Exception as exc:  # noqa: BLE001
            print(f"  atlandi: {exc}", flush=True)
            continue
        if not lines:
            print(f"  ornek bulunamadi", flush=True)
            continue
        path = out_dir / filename
        write_corpus(path, lines, description)
        print(f"  {len(lines)} ornek -> {path}", flush=True)
        manifest.append(f"{filename}|{len(lines)}")
        total += len(lines)

    manifest_path = out_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Toplam benzersiz indirilen ornek: {total}", flush=True)
    print(f"Benzersiz prompt sayisi (dedup): {len(seen)}", flush=True)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    count = download_all(args.out_dir, args.seed)
    return 0 if count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
