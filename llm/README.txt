================================================================================
LLM KLASORU - GENEL BAKIS
================================================================================

Bu klasor, projenin cekirdek yapay zeka kutuphanesini icerir.
Uygulama (CMakeProject2) bu kutuphaneyi kullanir; model mantigi burada yasar.

Simdiki durum: Egitim dongusu tamamlandi.
Backward + AdamW + Trainer mevcut.


================================================================================
KLASOR YAPISI
================================================================================

llm/
  CMakeLists.txt       -> llm_core statik kutuphanesini tanimlar
  README.txt           -> Bu dosya
  include/llm/         -> Disariya acik header dosyalari (public API)
  src/core/            -> .cpp implementasyon dosyalari


================================================================================
BUILD (CMakeLists.txt)
================================================================================

- llm_core adinda STATIK kutuphane uretilir.
- Derlenen kaynaklar: src/core/logger.cpp, src/core/config.cpp
- Diger moduller su an header-only (sadece .hpp).
- C++20 zorunludur.
- MSVC derleyicisinde /arch:AVX2 aciktir (ileride SIMD tensor islemleri icin).

Kullanim (baska hedeflerden):
  target_link_libraries(HedefAdi PRIVATE llm_core)


================================================================================
NE VAR, NE YOK?
================================================================================

Mevcut (Adim 1-2):
  - Temel tipler (Scalar, Index, TokenId)
  - Hata yonetimi (Status, Result<T>)
  - Logger
  - Model yapilandirmasi (ModelConfig, dosya okuma/yazma)
  - Hizali bellek (AlignedBuffer)
  - Surum bilgisi
  - Tensor, Shape, stride
  - MatMul (AVX2), Add/Sub/Mul, aktivasyonlar, Softmax
  - Vocabulary, BPE tokenizer, encode/decode, save/load
  - RMSNorm, RoPE, Multi-Head Attention (GQA), SwiGLU FFN
  - TransformerBlock, TransformerModel forward pass
  - Binary checkpoint save/load (model.ckpt)
  - KV-cache, InferenceEngine (Prefill / Decode)
  - Sampling, TextGenerator (temperature, top-k, top-p)
  - CLI/Chat (ChatSession, ChatCli)
  - INT8 quantization (.ckptq)
  - Batch inference (BatchInferenceEngine, BatchTextGenerator)
  - Egitim (Trainer, AdamW, cross-entropy backward)

Henuz yok (opsiyonel):
  - Runtime INT8 matmul (simdilik load'ta dequantize)
  - Padded batch prefill (tek pass)


================================================================================
GELISTIRME YOL HARITASI
================================================================================

Adim 1  Temel altyapi          (foundation)     TAMAMLANDI
Adim 2  Tensor motoru          (tensor)         TAMAMLANDI
Adim 3  Tokenizer              (tokenizer)      TAMAMLANDI
Adim 4  Model mimarisi         (model)          TAMAMLANDI
Adim 5  Agirlik yonetimi       (checkpoint)     TAMAMLANDI
Adim 6  Inference motoru       (inference)      TAMAMLANDI
Adim 7  Sampling               (sampling)       TAMAMLANDI
Adim 8  CLI / Chat             (cli)            TAMAMLANDI
Adim 9  Quantization           (quant)          TAMAMLANDI
Adim 10 Batch inference        (batch)          TAMAMLANDI
Adim 11 Egitim                 (training)       TAMAMLANDI
Adim 9  Quantization           FP16/INT8 inference
Adim 10 CLI / API              interaktif chat, model komutlari


================================================================================
INCLUDE / SRC AYRIMI
================================================================================

include/llm/  -> Ne kullanacagini soyler (public API)
src/core/     -> Nasil calistigini yazar (.cpp)

Bu ayrim buyudukce derlemeyi hizlandirir ve kodu okunakli tutar.
Detaylar icin alt klasorlerdeki README.txt dosyalarina bakin.

================================================================================
