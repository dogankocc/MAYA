================================================================================
src/inference - INFERENCE IMPLEMENTASYONLARI
================================================================================

kv_cache.cpp
  Katman basina onceden ayrilmis K/V bufferlari.
  Write(layer, position, keys, values) ile cache'e yazar.


inference_engine.cpp
  Prefill: prompt'u isler, cache doldurur, son token logits dondurur.
  Decode: tek token ekler, cache'i genisletir.
  ForwardHidden: tum katmanlarda ForwardWithCache cagirir.

================================================================================
