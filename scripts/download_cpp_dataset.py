#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C/C++ Egitim Veri Seti Indirici
Magicoder-Evol-Instruct-75K -> C/C++ filtreleme -> Proje JSONL formatina
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Set, List

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "corpus"

# C ve C++ anahtar kelimeleri
CPP_KEYWORDS = [
    "c++", "cpp", "c++11", "c++14", "c++17", "c++20", "c++23",
    "std::", "#include", "using namespace", "class ", "struct ",
    "vector<", "map<", "unordered_map", "iostream", "fstream",
    "printf", "scanf", "malloc", "free", "int main(",
    "c program", "c kod", "c++ program", "c++ kod",
    "```cpp", "```c++", "```c"
]


def is_cpp_example(instruction: str, response: str) -> bool:
    text = (instruction + " " + response).lower()
    for kw in CPP_KEYWORDS:
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


def ensure_datasets() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("[!] datasets kutuphanesi yuklu degil, yukleniyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  C/C++ EGITIM VERI SETI INDIRICI")
    print("=" * 60)
    print()

    ensure_datasets()

    from datasets import load_dataset

    print("[1/3] Magicoder-Evol-Instruct-75K yukleniyor...")
    print("      Bu islem internet hizina gore birkac dakika surebilir.")
    print()

    ds = load_dataset(
        "ise-uiuc/Magicoder-Evol-Instruct-75K",
        split="train",
    )

    print(f"      Toplam ornek sayisi: {len(ds)}")
    print()
    print("[2/3] C/C++ ornekleri filtreleniyor...")

    rng = random.Random(42)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    seen: Set[str] = set()
    lines: List[str] = []
    cpp_count = 0

    for i, idx in enumerate(indices):
        if len(lines) >= 50000:  # Maks 50 bin ornek
            break

        if (i + 1) % 10000 == 0:
            print(f"      {i+1}/{len(ds)} tarandi, C/C++ bulundu: {cpp_count}")

        item = ds[idx]
        instruction = item.get("instruction", "")
        response = item.get("response", "")

        if is_cpp_example(instruction, response):
            key = instruction.casefold()
            if key not in seen:
                seen.add(key)
                lines.append(format_jsonl_line(instruction, response))
                cpp_count += 1

    if cpp_count == 0:
        print()
        print("[HATA] C/C++ ornegi bulunamadi!")
        return 1

    print()
    print(f"[3/3] {cpp_count} C/C++ ornegi kaydediliyor...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUT_DIR / "cpp_code_magicoder.jsonl"

    with output_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Magicoder-Evol-Instruct-75K - C/C++ Filtreli ({cpp_count} ornek)\n")
        f.write("# Kod yazma, kod tamamlama, hata ayiklama, algoritma cozumu\n")
        for line in lines:
            f.write(line + "\n")

    file_size = output_file.stat().st_size
    size_mb = file_size / (1024 * 1024)

    print()
    print("=" * 60)
    print("  BASARILI!")
    print("=" * 60)
    print()
    print(f"Kaydedilen dosya: {output_file}")
    print(f"Toplam C/C++ ornegi: {cpp_count}")
    print(f"Dosya boyutu: {size_mb:.2f} MB")
    print()
    print("=" * 60)
    print("  EGITIMI BASLATMAK ICIN:")
    print("=" * 60)
    print()
    print("Method 1: Visual Studio")
    print("  - MAYA.cpp'i ac")
    print("  - Ctrl+F5 (Run without debugging)")
    print()
    print("Method 2: Komut satiri")
    print(f"  cd /d {ROOT}")
    print("  MAYA.exe train --corpus-dir data/corpus")
    print()
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
