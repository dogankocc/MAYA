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
    description: str
    instruction_key: str = "instruction"
    response_key: str = "response"
    input_key: Optional[str] = None
    split: str = "train"
    max_samples: int = MAX_SAMPLES_PER_DATASET
    language_filter: Optional[str] = None  # "cpp", "python", None=tumu

    @property
    def output_file(self) -> Path:
        return OUT_DIR / f"code_{self.name.lower().replace('-', '_')}.jsonl"


DATASETS = [
    DatasetInfo(
        "Magicoder-OSS",
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "Magicoder OSS-Instruct - C/C++ agirlikli kod uretimi",
        instruction_key="problem",
        response_key="solution",
        max_samples=30000,
        language_filter="cpp",
    ),
    DatasetInfo(
        "CodeAlpaca",
        "flwrlabs/code-alpaca-20k",
        "CodeAlpaca 20k - genel kod talimatlari",
        response_key="output",
        max_samples=20000,
    ),
    DatasetInfo(
        "Python-Instructions-18k",
        "iamtarun/python_code_instructions_18k_alpaca",
        "Python kod talimatlari (18k, alpaca formati)",
        response_key="output",
        input_key="input",
        max_samples=18000,
    ),
    DatasetInfo(
        "Evol-Instruct-Code-80k",
        "nickrosh/Evol-Instruct-Code-80k-v1",
        "Evol-Instruct Code 80k - zor ve cok adimli kod problemleri",
        response_key="output",
        max_samples=40000,
    ),
    DatasetInfo(
        "Glaive-Code-Assistant",
        "glaiveai/glaive-code-assistant",
        "Glaive Code Assistant - soru/cevap tarzi kod yardimi",
        instruction_key="question",
        response_key="answer",
        max_samples=40000,
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
            instruction = (item.get(info.instruction_key, "") or "").strip()
            response = (item.get(info.response_key, "") or "").strip()
            extra = (item.get(info.input_key, "") or "").strip() if info.input_key else ""
            if extra:
                instruction = f"{instruction}\n\n{extra}"

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


def write_dataset_file(info: DatasetInfo, lines: List[str]) -> None:
    random.Random(42).shuffle(lines)
    with info.output_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {info.description} ({len(lines)} ornek)\n")
        f.write(f"# Kaynak: {info.hf_id}\n")
        f.write("\n".join(lines) + "\n")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    force = "--force" in sys.argv[1:]

    print("=" * 70)
    print("  KAPSAMLI KOD EGITIM VERI SETI INDIRICI")
    print("=" * 70)
    print()
    print(f"  Hedef toplam: {MAX_TOTAL_SAMPLES} ornek")
    print(f"  Kaynak sayisi: {len(DATASETS)} farkli veri seti")
    print()

    ensure_dependencies()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seen: Set[str] = set()
    total = 0
    written: List[Path] = []

    for info in DATASETS:
        if total >= MAX_TOTAL_SAMPLES:
            break

        if info.output_file.exists() and not force:
            print(f"  [ATLA] {info.output_file.name} zaten var (yeniden indirmek icin --force)")
            continue

        lines = load_single_dataset(info, seen)
        if not lines:
            continue

        write_dataset_file(info, lines)
        written.append(info.output_file)
        total += len(lines)

    print()
    print("=" * 70)
    print(f"  TOPLAM: {total} benzersiz ornek, {len(written)} dosya")
    print("=" * 70)
    print()

    if total == 0:
        print("[HATA] Ornek yuklenemedi!")
        return 1

    for path in written:
        print(f"  Kaydedildi: {path} ({path.stat().st_size / (1024 * 1024):.2f} MB)")
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
