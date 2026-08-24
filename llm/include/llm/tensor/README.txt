================================================================================
include/llm/tensor - TENSOR MOTORU (Adim 2)
================================================================================

Cok boyutlu tensor veri yapisi, temel lineer cebir ve aktivasyon islemleri.
Namespace: llm (siniflar), llm::tensor (islemler)


================================================================================
DOSYALAR
================================================================================

shape.hpp
  Tensor boyutlarini ve row-major stride hesabini tutar.

  Shape{d0, d1, ...}   Boyut listesi
  Rank()               Eksen sayisi
  Numel()              Toplam eleman sayisi
  Strides()            Row-major (C-style) stride vektoru
  ToString()           Ornek: "(2, 3, 4)"

  Row-major: son eksen bellekte ardisik. Ornek [2,3] matris:
    indeks (row, col) -> offset = row * 3 + col


tensor.hpp
  Sayisal veri tasiyicisi.

  Tensor::Zeros(shape)       Sifir tensor
  Tensor::Ones(shape)        Birler tensor
  Tensor::FromBuffer(...)    Hazir veri ile olusturma
  At({i, j, ...})            Cok boyutlu indeks erisimi
  Reshape(newShape)          Ayni numel ile yeniden sekillendirme
  Transpose2D()              2D matris transpozu
  Fill(value)                Tum elemanlari doldurur

  Veri std::vector<Scalar> icinde tutulur (float32).


ops.hpp
  Tensor islemleri (llm::tensor namespace).

  MatMul(a, b, out)     2D matris carpimi (AVX2 SIMD)
  Add / Sub / Mul       Ayni shape, eleman bazli
  Scale(input, f, out)  Skaler carpim
  Relu(tensor)          max(0, x)
  Gelu(tensor)          Transformer FFN aktivasyonu
  Silu(tensor)          x * sigmoid(x)
  Softmax(tensor, axis) Rank-1 veya rank-2, eksen bazli


================================================================================
SIMD MATMUL
================================================================================

MatMul: C[M,N] = A[M,K] x B[K,N]

  - MSVC: /arch:AVX2
  - GCC/Clang: -mavx2 -mfma
  - 8'li float bloklar (_mm256), FMA ile hizlandirma
  - N % 8 != 0 kalan sutunlar scalar dongu ile

Kaynak: src/tensor/matmul.cpp


================================================================================
IMPLEMENTASYON
================================================================================

src/tensor/shape.cpp     Shape hesaplamalari
src/tensor/tensor.cpp    Tensor metotlari
src/tensor/matmul.cpp    AVX2 matmul cekirdegi
src/tensor/ops.cpp       Diger tensor islemleri


================================================================================
KULLANIM ORNEGI
================================================================================

  llm::Tensor a = llm::Tensor::FromBuffer(llm::Shape{2, 3}, {...});
  llm::Tensor b = llm::Tensor::FromBuffer(llm::Shape{3, 2}, {...});
  llm::Tensor c = llm::Tensor::Zeros(llm::Shape{2, 2});
  llm::tensor::MatMul(a, b, c);

  llm::Tensor logits = llm::Tensor::FromBuffer(llm::Shape{1, 4}, {...});
  llm::tensor::Softmax(logits, 1);


================================================================================
SONRAKI ADIMLARDA
================================================================================

  - Batched matmul (attention icin)
  - Broadcast kurallari
  - Weight tensor sabitleme / paylasimli bellek
  - FP16 / quantization destegi

================================================================================
