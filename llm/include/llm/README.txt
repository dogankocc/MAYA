================================================================================
include/llm - PUBLIC API
================================================================================

Bu klasor, llm_core kutuphanesinin disariya acik baslik dosyalarini icerir.
CMake, bu yolu otomatik olarak include path'e ekler.


================================================================================
DOSYALAR
================================================================================

llm.hpp
  Tek giris noktasi. Tum temel modulleri tek seferde dahil eder.
  Uygulama kodunda yeterli olan:

    #include "llm/llm.hpp"

  Boylece her header'i tek tek eklemek gerekmez.


version.hpp
  Proje surum ve asama bilgisi.

  llm::kVersionMajor   -> 0
  llm::kVersionMinor   -> 1
  llm::kVersionPatch   -> 0
  llm::kProjectName    -> "LLM"
  llm::kBuildStage     -> "foundation" (mevcut gelistirme asamasi)

  Loglama veya kullaniciya gosterilecek surum metni icin kullanilir.


core/
  Temel altyapi modulleri. Detayli aciklama:
  include/llm/core/README.txt


================================================================================
SONRAKI ADIMLARDA EKLENECEK KLASORLER (plan)
================================================================================

include/llm/tensor/      -> Tensor sinifi, shape, matris islemleri (Adim 2)
include/llm/tokenizer/   -> BPE tokenizer, vocabulary (Adim 3)
include/llm/model/       -> Transformer katmanlari, forward pass (Adim 4)
  model/checkpoint/      -> Binary checkpoint save/load (Adim 5)
include/llm/inference/   -> KV-cache, InferenceEngine (Adim 6)
  inference/sampling/  -> Sampler, TextGenerator (Adim 7)
include/llm/cli/         -> ChatSession, ChatCli (Adim 8)
include/llm/quantization/ -> INT8 quant checkpoint (Adim 9)

Her yeni modul eklendiginde llm.hpp guncellenerek tek giris noktasi korunur.

================================================================================
