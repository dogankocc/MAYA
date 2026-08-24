================================================================================
include/llm/model - TRANSFORMER MODEL (Adim 4)
================================================================================

GPT/Llama tarzi decoder-only Transformer mimarisi.
Namespace: llm::model


================================================================================
DOSYALAR
================================================================================

params.hpp
  Xavier agirlik baslatma yardimcilari.


linear.hpp
  Y = X @ W^T (+ bias)
  W shape: [output_dim, input_dim]
  X shape: [seq_len, input_dim]


rms_norm.hpp
  RMSNorm: x / RMS(x) * weight
  Llama tarzi normalizasyon (LayerNorm yerine).


rope.hpp
  RoPE (Rotary Position Embedding).
  RopeCache: cos/sin tablosu onceden hesaplanir.
  ApplyRope: Q ve K tensörlerine pozisyon kodlamasi uygular.


attention.hpp
  MultiHeadAttention:
    - Q projeksiyon: hidden -> hidden
    - K/V projeksiyon: hidden -> num_kv_heads * head_dim (GQA)
    - Causal (gelecegi gormeyen) masked attention
    - RoPE Q/K uzerinde
    - Output projeksiyon


ffn.hpp
  SwiGluFfn (Llama tarzi):
    gate = SiLU(x @ W_gate)
    up   = x @ W_up
    out  = (gate * up) @ W_down


transformer_block.hpp
  Pre-norm Transformer blogu:
    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))


transformer.hpp
  TransformerModel:
    - Token embedding [vocab, hidden]
    - N adet TransformerBlock
    - Final RMSNorm
    - LM head [vocab, hidden]
    - Forward(tokens) -> logits [seq, vocab]


================================================================================
MIMARI AKIS
================================================================================

  token IDs
      |
      v
  token embedding lookup
      |
      v
  +------------------+
  | TransformerBlock |  x num_layers
  |  RMSNorm         |
  |  Attention+RoPE  |
  |  RMSNorm         |
  |  SwiGLU FFN      |
  +------------------+
      |
      v
  final RMSNorm
      |
      v
  lm_head matmul
      |
      v
  logits [seq, vocab]


================================================================================
GQA (Grouped Query Attention)
================================================================================

  num_heads = 8, num_kv_heads = 2 orneginde:
    - 8 query head
    - 2 key/value head
    - Her 4 query head, 1 KV head paylasir

  Bellek ve hesaplama tasarrufu saglar.


================================================================================
IMPLEMENTASYON
================================================================================

src/model/
  params.cpp, linear.cpp, rms_norm.cpp, rope.cpp
  attention.cpp, ffn.cpp, transformer_block.cpp, transformer.cpp

checkpoint/ (Adim 5)
  include/llm/model/checkpoint/checkpoint.hpp
  src/model/checkpoint/checkpoint.cpp
  Detay: include/llm/model/checkpoint/README.txt


================================================================================
ORNEK
================================================================================

  llm::ModelConfig config = llm::Config::DefaultModelConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(42);
  model.ResetParameters(rng);

  std::vector<llm::TokenId> tokens = {10, 11, 12};
  llm::Tensor logits = llm::Tensor::Zeros(llm::Shape{tokens.size(), config.vocabSize});
  model.Forward(tokens, logits);


================================================================================
SONRAKI ADIMLARDA
================================================================================

  - KV-cache ile inference (Adim 6)
  - Batched forward
  - Checkpoint sikistirma

================================================================================
