================================================================================
LLM Chat UI - Android + Windows
================================================================================

Mimari:
  CMakeProject2 serve  ->  HTTP API (C++)
  chat_app (Flutter)   ->  Android + Windows arayuzu

-------------------------------------------------------------------------------
1) Model hazirla (ilk kez)
-------------------------------------------------------------------------------
  cd CMakeProject2
  out\build\x64-debug\CMakeProject2\CMakeProject2.exe demo

-------------------------------------------------------------------------------
2) Sunucuyu baslat (Windows)
-------------------------------------------------------------------------------
  CMakeProject2.exe serve --model model.ckptq --tokenizer tokenizer_data --port 8765

  Endpointler:
    GET  /api/v1/health
    POST /api/v1/chat   {"prompt":"...","max_tokens":64,"temperature":0.8}
    POST /api/v1/reset

-------------------------------------------------------------------------------
3) Windows arayuz
-------------------------------------------------------------------------------
  scripts\run_chat_windows.bat
  veya:
    cd chat_app
    flutter pub get
    flutter run -d windows

  Ayarlar > Sunucu: http://127.0.0.1:8765

-------------------------------------------------------------------------------
4) Android arayuz
-------------------------------------------------------------------------------
  PC'de sunucu calisirken:
    Gercek cihaz: http://<PC_IP>:8765
    Emulator:     http://10.0.2.2:8765 (varsayilan)

    cd chat_app
    flutter pub get
    flutter run -d android

  Not: Sunucu su an Windows'ta calisir. Telefonda yerel inference icin
  ileride NDK/FFI entegrasyonu eklenebilir.

-------------------------------------------------------------------------------
Ozellikler
-------------------------------------------------------------------------------
  - Turkce / Ingilizce (i18n)
  - Acik / koyu / sistem temasi
  - Sunucu URL, max token, temperature, greedy ayarlari
  - Sohbet sifirlama

================================================================================
