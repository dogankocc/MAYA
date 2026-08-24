================================================================================
src/model - MODEL IMPLEMENTASYONLARI
================================================================================

linear.cpp        Dense katman, MatMul tabanli
rms_norm.cpp      Satir bazli RMS normalizasyon
rope.cpp          Kosin/sin cache ve 2D rotasyon
attention.cpp     Causal multi-head attention + GQA + RoPE
ffn.cpp           SwiGLU feed-forward
transformer_block.cpp  Pre-norm residual blok
transformer.cpp   Tam model: embedding, katmanlar, lm head

checkpoint/       Binary checkpoint save/load
                  Detay: src/model/checkpoint/README.txt

================================================================================
