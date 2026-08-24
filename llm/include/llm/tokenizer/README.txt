================================================================================
include/llm/tokenizer - TOKENIZER (Adim 3)
================================================================================

Metin <-> token ID donusumu. BPE (Byte Pair Encoding) algoritmasi.
Namespace: llm


================================================================================
DOSYALAR
================================================================================

vocabulary.hpp
  Token <-> ID sozlugu.

  Ozel tokenlar (sabit ID):
    0  <PAD>
    1  <UNK>
    2  <BOS>
    3  <EOS>

  AddToken(token)     Yeni token ekler, ID dondurur
  GetId(token)        ID bulur, yoksa <UNK>
  GetToken(id)        ID'den token metni
  Save / Load         vocab.txt (satir basina bir token, satir no = ID)


bpe_tokenizer.hpp
  BPE tokenizer: egitim, encode, decode.

  kWordStartToken = U+2581 (▁) kelime basi isareti (SentencePiece tarzi)

  Train(corpus, targetVocabSize)
    - Corpus'u kelimelere boler
    - Her kelime: [▁, c1, c2, ...] karakter dizisine ayrilir
    - En sik ciftleri birlestirerek merge kurallari ogrenir
    - Hedef vocab boyutuna ulasana kadar devam eder

  Encode(text)
    - Kelime kelime BPE uygular (kelimeler arasi merge yok)
    - Token ID listesi dondurur

  Decode(ids)
    - ID listesini metne cevirir
    - ▁ karakterini bosluga donusturur

  EncodeWithSpecialTokens(text, addBos, addEos)
    - BOS/EOS ekleyerek encode eder

  Save(directory) / Load(directory)
    - vocab.txt  : sozluk
    - merges.txt : merge kurallari (tab ile ayrilmis: left\tright)


================================================================================
BPE ENCODE AKISI
================================================================================

  "araba model"
       |
       v
  ["araba"] ["model"]           <- kelime ayirma
       |
       v
  [▁,a,r,a,b,a] [▁,m,o,d,e,l]  <- karakterlere bolme
       |
       v
  merge kurallari (oncelik sirasina gore)
       |
       v
  [▁araba] [▁model]             <- ornek sonuc (egitime bagli)
       |
       v
  [id1, id2]                    <- vocabulary'den ID


================================================================================
IMPLEMENTASYON
================================================================================

src/tokenizer/vocabulary.cpp
src/tokenizer/bpe_tokenizer.cpp


================================================================================
ORNEK KULLANIM
================================================================================

  llm::BpeTokenizer tokenizer;
  tokenizer.Train("araba araba model model", 64);

  auto ids = tokenizer.EncodeWithSpecialTokens("araba", true, true);
  std::string text = tokenizer.Decode(ids);

  tokenizer.Save("tokenizer_data");
  auto loaded = llm::BpeTokenizer::Load("tokenizer_data");


================================================================================
SONRAKI ADIMLARDA
================================================================================

  - Byte-level BPE (UTF-8 / cok dilli destek)
  - Onceden egitilmis vocab yukleme (GPT-2 tarzi)
  - Batch encode / padding mask

================================================================================
