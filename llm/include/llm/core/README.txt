================================================================================
include/llm/core - TEMEL ALTYAPI MODULLERI
================================================================================

Adim 1 kapsaminda tum cekirdek yardimci siniflar bu klasordedir.
Namespace: llm


================================================================================
1. types.hpp - Temel Tipler
================================================================================

Projede kullanilan ortak tip takma adlari ve sabitler.

  Scalar     = float
               Model agirliklari ve sayisal hesaplamalar icin.

  Index      = int64_t
               Tensor indeksleri ve negatif offset destegi icin.

  TokenId    = uint32_t
               Tokenizer tarafindan uretilen token kimlikleri icin.

  Dimension  = size_t
               Boyut (vocab_size, hidden_dim vb.) ifadeleri icin.

Sabitler:
  kInvalidToken    = UINT32_MAX   Gecersiz token ID
  kInvalidIndex    = -1           Gecersiz indeks
  kSimdAlignment   = 32           AVX2 icin bellek hizalama (byte)


================================================================================
2. status.hpp - Hata Yonetimi
================================================================================

Exception yerine acik hata kontrolu. Performansli ve okunakli.

ErrorCode enum:
  Ok, InvalidArgument, OutOfRange, IoError, NotImplemented, Internal

Status
  Deger dondurmeyen islemler icin.
  Ornek:
    llm::Status s = llm::Config::SaveToFile(config, "model.conf");
    if (!s.IsOk()) { /* s.Message() */ }

Result<T>
  Deger donduren islemler icin.
  Ornek:
    auto result = llm::Config::LoadFromFile("model.conf");
    if (result.IsOk()) {
        const llm::ModelConfig& cfg = result.Value();
    } else {
        /* result.GetError().message */
    }


================================================================================
3. logger.hpp - Loglama
================================================================================

Thread-safe, seviyeli logger. Uygulama genelinde tutarli cikti icin.

Log seviyeleri (dusukten yuksege):
  Trace, Debug, Info, Warn, Error

Kullanim:
  auto& log = llm::Logger::Instance();
  log.SetLevel(llm::LogLevel::Info);
  log.SetOutput(std::cout);          // varsayilan: std::cerr
  log.Info("bilesen_adi", "mesaj");

Cikti formati:
  YYYY-MM-DD HH:MM:SS.mmm [SEVIYE] [bilesen] mesaj

logger.cpp implementasyonu: src/core/logger.cpp


================================================================================
4. config.hpp - Model Yapilandirmasi
================================================================================

Transformer model hiperparametrelerini tutar ve dosyadan okur/yazar.

ModelConfig alanlari (varsayilan degerler):

  vocabSize         32000    Kelime dagarcigi boyutu
  hiddenDim         512      Gizli katman boyutu (d_model)
  numLayers         6        Transformer blok sayisi
  numHeads          8        Multi-head attention head sayisi
  numKvHeads        8        KV-cache head sayisi (GQA icin)
  intermediateDim   2048     Feed-forward ara katman (FFN)
  maxSeqLen         2048     Maksimum sequence uzunlugu
  ropeTheta         10000    RoPE pozisyon kodlama tabani
  normEps           1e-5     Layer normalization epsilon

Config sinifi:
  DefaultModelConfig()           Varsayilan ModelConfig dondurur
  LoadFromFile(path)             Key=value dosyasindan okur, Validate eder
  SaveToFile(config, path)       Dosyaya yazar

ModelConfig::Validate() kontrolleri:
  - vocabSize, hiddenDim, numLayers, numHeads, numKvHeads > 0
  - hiddenDim % numHeads == 0
  - numHeads % numKvHeads == 0  (Grouped Query Attention uyumu)
  - maxSeqLen > 0
  - ropeTheta > 0, normEps > 0

Ornek config dosyasi: proje kokundeki config/model.conf

config.cpp implementasyonu: src/core/config.cpp


================================================================================
5. memory.hpp - Hizali Bellek
================================================================================

SIMD (AVX2) islemleri icin 32-byte hizali dinamik dizi sablonu.

AlignedBuffer<T>
  - Resize(count)     Eleman sayisina gore hizali bellek ayirir
  - Data()            Ham pointer
  - Size()            Eleman sayisi
  - operator[]        Indeks erisimi
  - Move semantics    Tasima destekler; kopyalama yasak

Ornek:
  llm::AlignedBuffer<llm::Scalar> buffer(1024);
  buffer[0] = 1.0f;

Ileride tensor verisi bu sinif uzerinden veya benzeri bir yapida tutulacak.


================================================================================
BAGIMLILIK OZETI
================================================================================

  types.hpp    -> bagimsiz
  status.hpp   -> bagimsiz
  memory.hpp   -> types.hpp
  config.hpp   -> status.hpp
  logger.hpp   -> bagimsiz

================================================================================
