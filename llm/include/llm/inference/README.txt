================================================================================
include/llm/inference - INFERENCE MOTORU (Adim 6)
================================================================================

KV-cache ile verimli autoregressive inference.
Namespace: llm::inference


================================================================================
DOSYALAR
================================================================================

kv_cache.hpp
  Katman basina K/V tensorlerini tutar.
  Shape: [max_seq_len, kv_dim] per layer
  kv_dim = num_kv_heads * head_dim


batch_kv_cache.hpp
  Batch boyutlu KV-cache: [batch, max_seq_len, kv_dim]

batch_inference_engine.hpp
  PrefillBatch / DecodeBatch ile coklu sequence inference

inference_engine.hpp
  Tek sequence prefill + decode (batch=1)


================================================================================
KV-CACHE
================================================================================

Her Transformer katmani icin:
  - keys[layer]   : [max_seq_len, kv_dim]
  - values[layer] : [max_seq_len, kv_dim]
  - length_       : su ana kadar yazilan token sayisi

Prefill: tum prompt K/V cache'e yazilir.
Decode:  her yeni token icin yalnizca 1 satir K/V eklenir;
         attention onceki cache'e bakar (tekrar hesaplamaz).


================================================================================
INFERENCE AKISI
================================================================================

Prefill(prompt_tokens):
  1. Cache sifirla
  2. Token embedding -> [seq, hidden]
  3. Katmanlar ForwardWithCache (cacheStart=0)
  4. Son token hidden -> RMSNorm -> lm_head
  5. logits [1, vocab]

Decode(token):
  1. Tek token embed -> [1, hidden]
  2. Katmanlar ForwardWithCache (cacheStart=mevcut uzunluk)
  3. logits [1, vocab]
  4. cache length += 1


================================================================================
MODEL DEGISIKLIKLERI
================================================================================

  MultiHeadAttention::ForwardWithCache
  TransformerBlock::ForwardWithCache
  ApplyRope(..., positionOffset)  -> decode'da dogru pozisyon

  Tam sequence Forward() hala mevcut (egitim / karsilastirma icin).


================================================================================
BATCH INFERENCE (Adim 10)
================================================================================

BatchInferenceEngine(model, batchSize):
  PrefillBatch(prompts)  -> logits [B, vocab]
  DecodeBatch(tokens)    -> logits [B, vocab]
  SetActive(mask)        -> biten sequence'leri decode'dan cikar

BatchKvCache:
  keys[layer]   : [B, max_seq_len, kv_dim]
  values[layer] : [B, max_seq_len, kv_dim]
  lengths_[b]   : her slot icin ayri uzunluk

BatchTextGenerator:
  GenerateBatch(prompts, maxNewTokens) -> vector<vector<TokenId>>

Prefill: her prompt ayri slot'ta (degisken uzunluk destekli).
Decode: tum aktif slot'lar tek matmul ile [B, hidden] islenir.


================================================================================
ORNEK (tek sequence)
================================================================================

  llm::inference::InferenceEngine engine(model);

  llm::Tensor logits = llm::Tensor::Zeros(llm::Shape{1, config.vocabSize});
  engine.Prefill({10, 11, 12}, logits);

  engine.Decode(13, logits);  // sonraki token logits


================================================================================
ORNEK (batch)
================================================================================

  const std::size_t batchSize = 2;
  llm::inference::BatchInferenceEngine batchEngine(model, batchSize);
  llm::inference::BatchTextGenerator batchGen(batchEngine);

  const std::vector<std::vector<llm::TokenId>> prompts = {{10, 11}, {20, 21, 22}};
  auto sequences = batchGen.GenerateBatch(prompts, 8, rng);


================================================================================
IMPLEMENTASYON
================================================================================

  src/inference/kv_cache.cpp
  src/inference/inference_engine.cpp
  src/inference/batch_kv_cache.cpp
  src/inference/batch_inference_engine.cpp
  Detay: src/inference/README.txt

sampling/ (Adim 7)
  Sampler, TextGenerator - temperature, top-k, top-p
  Detay: include/llm/inference/sampling/README.txt


================================================================================
SONRAKI ADIMLARDA
================================================================================

  - Padded prefill (tek forward pass, B>1)
  - Continuous batching
  - Repetition penalty

================================================================================
