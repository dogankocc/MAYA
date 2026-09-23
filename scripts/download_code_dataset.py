#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kapsamli Kod Egitim Veri Seti Indirici
Birden fazla kaynaktan kod egitim verisi indirir ve birlestirir
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Set, List, Optional, Callable, Any

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "corpus"

MAX_SAMPLES_PER_DATASET = 50000
MAX_TOTAL_SAMPLES = 200000

LANG_KEYWORDS = {
    "python": ["python", "def ", "import ", "from ", "__init__", "class ", "print(", "pandas", "numpy", "torch.", "tf."],
    "cpp": ["c++", "cpp", "std::", "#include", "using namespace", "class ", "struct ", "vector<", "map<", "int main(", "printf", "scanf", "malloc", "cout", "cin"],
    "c": ["#include", "int main(", "printf", "scanf", "malloc", "free", "struct ", "typedef "],
    "javascript": ["javascript", "js", "function ", "const ", "let ", "var ", "=>", "async ", "await ", "document.", "console."],
    "typescript": ["typescript", "ts", "interface ", "type ", "const ", ": string", ": number", ": boolean"],
    "csharp": ["c#", "csharp", "using ", "namespace ", "class ", "public ", "private ", "void ", "static "],
    "java": ["java", "public class", "private ", "void ", "static ", "String ", "System.", "ArrayList", "HashMap"],
    "go": ["golang", "package ", "import ", "func ", "var ", "const ", "type ", "struct ", "interface ", "go func"],
    "rust": ["rust", "fn ", "let ", "mut ", "const ", "struct ", "enum ", "impl ", "pub ", "use ", "crate::"],
    "sql": ["sql", "SELECT ", "FROM ", "WHERE ", "JOIN ", "INSERT ", "UPDATE ", "DELETE ", "CREATE TABLE", "INDEX"],
}


@dataclass
class DatasetInfo:
    name: str
    hf_id: str
    instruction_key: str = "instruction"
    response_key: str = "response"
    split: str = "train"
    max_samples: int = MAX_SAMPLES_PER_DATASET
    language_filter: Optional[str] = None  # "cpp", "python", None=tumu


DATASETS = [
    DatasetInfo(
        "Magicoder-OSS",
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        instruction_key="instruction",
        response_key="response",
        max_samples=50000,
        language_filter="cpp"
    ),
    DatasetInfo(
        "CodeAlpaca",
        "flwrlabs/code-alpaca-20k",
        instruction_key="instruction",
        response_key="output",
        max_samples=20000,
        language_filter=None
    ),
]


def ensure_dependencies() -> None:
    missing = []
    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")

    if missing:
        print(f"[!] Eksik kutuphaneler: {missing}")
        print("[!] Yukleniyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing + ["-q"])
        print("[!] Tamamlandi")


def matches_language(instruction: str, response: str, language: Optional[str]) -> bool:
    if language is None:
        return True

    text = (instruction + " " + response).lower()
    keywords = LANG_KEYWORDS.get(language, [])

    for kw in keywords:
        if kw.lower() in text:
            return True
    return False


def make_messages_record(user: str, assistant: str) -> dict:
    return {
        "intent": "code_generation",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def format_jsonl_line(user: str, assistant: str) -> str:
    return json.dumps(
        make_messages_record(user, assistant),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def load_single_dataset(info: DatasetInfo, seen: Set[str]) -> List[str]:
    from datasets import load_dataset

    print()
    print(f"  [=> Yukleniyor: {info.name} ({info.hf_id})]")

    try:
        ds = load_dataset(
            info.hf_id,
            split=info.split,
        )
    except Exception as e:
        print(f"     [HATA] Yuklenemedi: {e}")
        return []

    print(f"     Toplam: {len(ds)} ornek")

    rng = random.Random(42)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    lines: List[str] = []
    count = 0
    filtered_out = 0

    for i, idx in enumerate(indices):
        if len(lines) >= info.max_samples:
            break

        if (i + 1) % 10000 == 0:
            print(f"     {i+1}/{len(ds)} tarandi, eklendi: {count}, filtrelendi: {filtered_out}")

        try:
            item = ds[idx]
            instruction = item.get(info.instruction_key, "") or ""
            response = item.get(info.response_key, "") or ""

            if not instruction or not response:
                continue

            if not matches_language(instruction, response, info.language_filter):
                filtered_out += 1
                continue

            key = instruction.casefold()
            if key not in seen:
                seen.add(key)
                lines.append(format_jsonl_line(instruction, response))
                count += 1
        except Exception:
            continue

    print(f"     Tamamlandi: {count} yeni ornek eklendi")
    if info.language_filter:
        print(f"     Dil filtresi: {info.language_filter}")

    return lines


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  KAPSAMLI KOD EGITIM VERI SETI INDIRICI")
    print("=" * 70)
    print()
    print(f"  Hedef toplam: {MAX_TOTAL_SAMPLES} ornek")
    print(f"  Kaynak sayisi: {len(DATASETS)} farkli veri seti")
    print()

    ensure_dependencies()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_lines: List[str] = []
    seen: Set[str] = set()

    for info in DATASETS:
        if len(all_lines) >= MAX_TOTAL_SAMPLES:
            break

        lines = load_single_dataset(info, seen)
        all_lines.extend(lines)

    print()
    print("=" * 70)
    print(f"  TOPLAM: {len(all_lines)} benzersiz ornek")
    print("=" * 70)
    print()

    if len(all_lines) == 0:
        print("[HATA] Ornek yuklenemedi!")
        return 1

    random.Random(42).shuffle(all_lines)

    output_file = OUT_DIR / "code_training_unified.jsonl"

    with output_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Birlestirilmis kod egitim veri seti ({len(all_lines)} ornek)\n")
        f.write(f"# Kaynaklar: {', '.join(d.name for d in DATASETS)}\n")
        for line in all_lines:
            f.write(line + "\n")

    file_size = output_file.stat().st_size
    size_mb = file_size / (1024 * 1024)

    print(f"  Kaydedildi: {output_file}")
    print(f"  Dosya boyutu: {size_mb:.2f} MB")
    print()
    print("=" * 70)
    print("  EGITIMI BASLATMAK ICIN:")
    print("=" * 70)
    print()
    print("  Visual Studio: MAYA.cpp -> Ctrl+F5")
    print()
    print("  Komut satiri:")
    print(f"    cd /d {ROOT}")
    print("    MAYA.exe train --corpus-dir data/corpus")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
