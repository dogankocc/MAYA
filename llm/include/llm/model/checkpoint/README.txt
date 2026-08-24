================================================================================
include/llm/model/checkpoint - AGIRLIK YONETIMI (Adim 5)
================================================================================

Binary checkpoint formati ile model agirliklarini kaydetme ve yukleme.
Namespace: llm::model


================================================================================
DOSYALAR
================================================================================

checkpoint.hpp
  Checkpoint::Save / Checkpoint::Load API


================================================================================
API
================================================================================

  Checkpoint::Save(model, "model.ckpt")
  Checkpoint::Load("model.ckpt") -> TransformerModel


================================================================================
DOSYA FORMATI (model.ckpt)
================================================================================

Header:
  magic[8]           "LLMCKPT1"
  version            uint32 (su an: 1)
  vocab_size         uint64
  hidden_dim         uint64
  num_layers         uint64
  num_heads          uint64
  num_kv_heads       uint64
  intermediate_dim   uint64
  max_seq_len        uint64
  rope_theta         float32
  norm_eps           float32
  num_tensors        uint32

Her tensor:
  name_len           uint32
  name               UTF-8 string
  rank               uint32
  dims[rank]         uint64 each
  numel              uint64
  data               float32[numel]


================================================================================
TENSOR ISIMLENDIRME
================================================================================

  token_embedding
  final_norm_weight
  lm_head_weight
  layer.{i}.attn_norm_weight
  layer.{i}.ffn_norm_weight
  layer.{i}.attn.q_weight
  layer.{i}.attn.k_weight
  layer.{i}.attn.v_weight
  layer.{i}.attn.o_weight
  layer.{i}.ffn.gate_weight
  layer.{i}.ffn.up_weight
  layer.{i}.ffn.down_weight


================================================================================
DOGRULAMA
================================================================================

  - Magic ve version kontrolu
  - ModelConfig Validate()
  - Her tensor shape eslesmesi
  - Eksik tensor reddedilir


================================================================================
ORNEK
================================================================================

  llm::model::TransformerModel model(config);
  model.ResetParameters(rng);
  llm::model::Checkpoint::Save(model, "model.ckpt");

  auto loaded = llm::model::Checkpoint::Load("model.ckpt");
  loaded.Value().Forward(tokens, logits);


================================================================================
IMPLEMENTASYON
================================================================================

  src/model/checkpoint/checkpoint.cpp
  Detay: src/model/checkpoint/README.txt


================================================================================
SONRAKI ADIMLARDA
================================================================================

  - Sikistirma (gzip/zstd)
  - Incremental / shard checkpoint
  - Optimizer state (egitim icin)

================================================================================
