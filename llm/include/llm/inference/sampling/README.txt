================================================================================
include/llm/inference/sampling - SAMPLING (Adim 7)
================================================================================

Logits'ten sonraki token secimi ve metin uretim dongusu.
Namespace: llm::inference


================================================================================
DOSYALAR
================================================================================

sampler.hpp
  SamplingConfig, Sampler

generator.hpp
  TextGenerator - prefill + decode + sample dongusu


================================================================================
SamplingConfig
================================================================================

  temperature   Logit olcekleme (1.0 = degisiklik yok, dusuk = daha deterministik)
  topK          0 = kapali; yalnizca en yuksek K aday
  topP          Nucleus sampling (0-1); olasilik kumulatif esigi
  greedy        true ise argmax
  eosTokenId    Uretimi durduran token (varsayilan <EOS>)


================================================================================
Sampler
================================================================================

  SampleGreedy(logits)              En yuksek logit
  Sample(logits, config, rng)       Tam pipeline:
    1. temperature uygula
    2. top-k filtrele
    3. softmax
    4. top-p filtrele
    5. multinomial ornekleme

  logits shape: [1, vocab_size]


================================================================================
TextGenerator
================================================================================

  Generate(prompt, maxNewTokens, rng):
    1. Prefill(prompt)
    2. Dongu (maxNewTokens):
         token = Sample(logits)
         sequence'e ekle
         EOS ise dur
         max_seq_len asildiysa dur
         Decode(token) -> yeni logits
    3. prompt + uretilen tokenlar dondurur


================================================================================
ORNEK
================================================================================

  llm::inference::InferenceEngine engine(model);
  llm::inference::SamplingConfig config;
  config.temperature = 0.8f;
  config.topK = 40;
  config.topP = 0.9f;

  llm::inference::TextGenerator generator(engine, config);
  std::mt19937 rng(42);
  auto tokens = generator.Generate({10, 11, 12}, 32, rng);


================================================================================
IMPLEMENTASYON
================================================================================

  src/inference/sampling/sampler.cpp
  src/inference/sampling/generator.cpp
  Detay: src/inference/sampling/README.txt

================================================================================
