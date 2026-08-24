================================================================================
include/llm/quantization - QUANTIZATION (Adim 9)
================================================================================

INT8 symmetric quantization ile model sikistirma.
Namespace: llm::quantization


================================================================================
DOSYALAR
================================================================================

quantize.hpp
  QuantizeSymmetric, Dequantize, MaxAbsError

quant_checkpoint.hpp
  QuantCheckpoint::Save / Load / ConvertFile


================================================================================
ALGORITMA (symmetric INT8)
================================================================================

  scale = max(abs(tensor)) / 127
  int8  = round(value / scale)   [-127, 127]

  Dequantize:
  float = int8 * scale


================================================================================
DOSYA FORMATI (.ckptq)
================================================================================

  magic: "LLMQCKPT"
  Header: model config (FP32 checkpoint ile ayni alanlar)

  Her tensor:
    name, shape, numel, scale (float32), int8_data[numel]

  Yukleme: INT8 -> dequantize -> FP32 model (mevcut forward ile uyumlu)


================================================================================
KULLANIM
================================================================================

  QuantCheckpoint::ConvertFile("model.ckpt", "model.ckptq");
  auto model = QuantCheckpoint::Load("model.ckptq");

  MAYA quantize --input model.ckpt --output model.ckptq


================================================================================
BELLEK KAZANCI
================================================================================

  Agirliklar ~4x daha kucuk (float32 -> int8 + scale)
  Inference simdilik FP32 (yukleme sirasinda dequantize)


================================================================================
IMPLEMENTASYON
================================================================================

  src/quantization/quantize.cpp
  src/quantization/quant_checkpoint.cpp

================================================================================
