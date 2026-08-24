================================================================================
src/tensor - TENSOR IMPLEMENTASYONLARI
================================================================================

shape.cpp
  Shape::ComputeStrides, Numel ve ToString implementasyonu.


tensor.cpp
  Tensor fabrika metotlari, At/Reshape/Transpose2D/Fill.


matmul.cpp
  AVX2 SIMD matris carpimi cekirdegi.
  matmul_internal.hpp: MatMulAvx2 / MatMulScalar bildirimleri.
  AVX2 yoksa otomatik scalar fallback.


ops.cpp
  MatMul sarmalayici, eleman bazli islemler, aktivasyonlar, softmax.
  Gelu: tanh tabanli yaklasim (Transformer uyumlu).
  Softmax: numerik stabil (max cikarma), rank-1 ve rank-2 destek.

================================================================================
