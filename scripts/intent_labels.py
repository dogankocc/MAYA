"""Intent labels and heuristic classification for Turkish SFT corpora."""

from __future__ import annotations

import re
from typing import Iterable

KNOWN_INTENTS = (
    "greeting",
    "farewell",
    "gratitude",
    "self_intro",
    "capability",
    "technical",
    "coding",
    "reasoning",
    "personal_profile",
    "wellbeing",
    "advice",
    "creative",
    "project_help",
    "general",
)


def normalize_intent(value: str) -> str:
    cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
    return cleaned if cleaned in KNOWN_INTENTS else "general"


def classify_intent(user: str, assistant: str = "") -> str:
    text = f"{user} {assistant}".lower()

    if any(token in text for token in ("merhaba", "selam", "gunaydin", "iyi aksam", "naber")):
        if any(token in text for token in ("kimsin", "kendini", "tanit", "sen kimsin")):
            return "self_intro"
        return "greeting"
    if any(token in text for token in ("gorusuruz", "hosca kal", "bay bay", "gule gule")):
        return "farewell"
    if any(token in text for token in ("tesekkur", "sagol", "eyvallah", "cok oldu")):
        return "gratitude"
    if any(token in text for token in ("kimsin", "kendini", "tanit", "sen kimsin", "ne yapabilirsin")):
        return "self_intro"
    if any(token in text for token in ("kod", "debug", "hata aliyorum", "compile", "cmake", "flutter", "react native", "expo")):
        return "coding"
    if any(token in text for token in ("adim adim", "mantik", "hesapla", "cozumle", "kanit", "sonuc cikar", "ortalama hiz")):
        return "reasoning"
    if any(
        token in text
        for token in (
            "python",
            "api",
            "llm",
            "transformer",
            "tokenizer",
            "sinir ag",
            "machine learning",
            "deep learning",
            "algoritma",
            "veritaban",
            "docker",
            "kubernetes",
        )
    ):
        return "technical"
    if any(token in text for token in ("yorgun", "uzgun", "stres", "mutlu", "sikildim", "uyuyamiyorum")):
        return "wellbeing"
    if any(token in text for token in ("oner", "tavsiye", "plan yap", "kariyer", "ogrenmeliyim", "hangi dili")):
        return "advice"
    if any(token in text for token in ("hikaye", "siir", "espri")):
        return "creative"
    if any(token in text for token in ("proje", "modeli nasil", "egitim", "sunucu", "mobil")):
        return "project_help"
    if any(
        token in text
        for token in (
            "gelistirici",
            "mikroservis",
            "gomulu",
            "otomobil",
            "ogrenme tarzi",
            "calisma ortami",
            "hedeflerin",
            "yapay zekada en cok",
        )
    ):
        return "personal_profile"
    return "general"


def profile_intent_for_prompt(user: str) -> str | None:
    text = user.lower()
    mapping = (
        ("kendini nasil tanimlarsin", "self_intro"),
        ("yeni bir teknolojiye nasil yaklasirsin", "personal_profile"),
        ("yapay zekada en cok neye ilgi", "technical"),
        ("kod yazma tarzin", "coding"),
        ("bir problemi nasil cozersin", "reasoning"),
        ("c++ ile calisma yaklasimin", "technical"),
        ("hedeflerin neler", "personal_profile"),
        ("otomobil projelerine yaklasimin", "personal_profile"),
        ("ogrenme tarzin", "advice"),
        ("sana en uygun calisma ortami", "personal_profile"),
    )
    for needle, intent in mapping:
        if needle in text:
            return intent
    return None
