================================================================================
src/model/checkpoint - CHECKPOINT IMPLEMENTASYONU
================================================================================

checkpoint.cpp
  Binary serilestirme ve model agirliklarinin geri yuklenmesi.

  Save akisi:
    1. ModelConfig + tensor sayisini header'a yaz
    2. Tum agirlik tensorlerini isim + shape + float32 veri olarak yaz

  Load akisi:
    1. Header dogrula (magic, version, config)
    2. ModelConfig'ten TransformerModel olustur
    3. Tensorleri isimle eslestir (AssignTensor)
    4. Eksik tensor kontrolu

  Public API: include/llm/model/checkpoint/checkpoint.hpp
  Format detayi: include/llm/model/checkpoint/README.txt

================================================================================
