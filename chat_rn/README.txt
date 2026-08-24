================================================================================
LLM Chat - React Native + Ollama (ChatGPT-class responses)
================================================================================

Mimari:
  Ollama (llama3.2 vb.)  ->  gercek buyuk dil modeli
  CMakeProject2 serve    ->  HTTP proxy (OpenAI uyumlu API)
  chat_rn                ->  Android arayuzu

-------------------------------------------------------------------------------
1) Ollama kur
-------------------------------------------------------------------------------
  https://ollama.com

  Model indir:
    ollama pull llama3.2

  (Alternatif: llama3.1, mistral, gemma2)

-------------------------------------------------------------------------------
2) Sunucuyu baslat
-------------------------------------------------------------------------------
  scripts\run_server.bat

  Varsayilan:
    Backend: openai (Ollama)
    Upstream: http://127.0.0.1:11434
    Model: llama3.2

  Eski mini model icin:
    scripts\run_server_local.bat

-------------------------------------------------------------------------------
3) React Native uygulama
-------------------------------------------------------------------------------
  cd chat_rn
  npx expo start --android

  Sunucu adresi:
    Emulator: http://10.0.2.2:8765
    Gercek cihaz: http://<PC_IP>:8765

-------------------------------------------------------------------------------
Neden iki backend?
-------------------------------------------------------------------------------
  openai  -> Ollama/LM Studio ile ChatGPT seviyesine yakin cevaplar
  local   -> Sifirdan egitilmis kucuk demo model (ogrenme amacli)

================================================================================
