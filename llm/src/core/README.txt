================================================================================
src/core - IMPLEMENTASYON DOSYALARI
================================================================================

Bu klasordeki .cpp dosyalari, include/llm/core/ altindaki header'larin
calisan kodunu icerir. llm_core statik kutuphanesine derlenir.


================================================================================
DOSYALAR
================================================================================

logger.cpp
  llm::Logger singleton implementasyonu.

  - SetLevel / SetOutput: calisma zamaninda log seviyesi ve cikis akisi
  - Log: zaman damgasi + seviye + bilesen adi + mesaj yazar
  - Thread-safe: std::mutex ile korunur
  - Windows: localtime_s; diger platformlar: localtime_r

  Header-only degildir cunku:
  - Zaman formatlama ve I/O islemleri derleme süresini uzatir
  - Tek bir .cpp'de toplanmasi link süresini ve yeniden derlemeyi iyilestirir


config.cpp
  llm::Config ve llm::ModelConfig::Validate implementasyonu.

  LoadFromFile:
  - Satir satir key=value formatini okur
  - # ile baslayan satirlari yorum sayar
  - Bilinmeyen anahtar veya gecersiz deger -> Result::Fail
  - Yukleme sonrasi Validate() cagrilir

  SaveToFile:
  - Tum ModelConfig alanlarini key=value satirlari olarak yazar

  Validate:
  - Transformer mimarisi icin tutarlilik kurallarini kontrol eder
  - Ornek: hidden_dim, num_heads'a tam bolunmeli

  Header-only degildir cunku:
  - Dosya I/O ve parse mantigi buyudukce .cpp'de tutmak daha okunaklidir


================================================================================
YENI MODUL EKLERKEN
================================================================================

1. include/llm/<modul>/ altina .hpp ekle
2. Gerekirse src/<modul>/ altina .cpp ekle
3. llm/CMakeLists.txt icindeki add_library(llm_core ...) listesine .cpp ekle
4. include/llm/llm.hpp veya ilgili ust header'a include ekle
5. tests/ altina unit test ekle

Sadece sablon veya kucuk inline fonksiyonlar iceren moduller header-only
kalabilir; I/O, agir hesaplama veya buyuk implementasyon .cpp'ye tasinmali.

================================================================================
