================================================================================
src/tokenizer - TOKENIZER IMPLEMENTASYONLARI
================================================================================

vocabulary.cpp
  Sozluk yonetimi ve vocab.txt okuma/yazma.


bpe_tokenizer.cpp
  BPE egitim dongusu:
    1. Corpus'tan kelime frekanslari
    2. Cift (bigram) sayimi
    3. En sik cifti birlestir, vocab'e ekle
    4. Hedef boyuta veya frekans < 2 olana kadar tekrarla

  Encode sirasinda mergeRank_ haritasi ile en dusuk rank'li (en once
  ogrenilen) merge secilir ve greedy uygulanir.

================================================================================
