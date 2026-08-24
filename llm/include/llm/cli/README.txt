================================================================================
include/llm/cli - CLI / CHAT (Adim 8)
================================================================================

Interaktif terminal sohbet arayuzu.
Namespace: llm::cli


================================================================================
DOSYALAR
================================================================================

chat_session.hpp
  Model + tokenizer + inference + sampling birlestirir.

chat_cli.hpp
  stdin/stdout uzerinde interaktif dongu.


================================================================================
ChatSession
================================================================================

  Load(modelPath, tokenizerPath)
    - FP32 veya quantized checkpoint (.ckpt / .ckptq)
    - BPE tokenizer dizini (vocab.txt + merges.txt)

  Complete(prompt, maxNewTokens, rng) -> string
    1. Prompt'u tokenize et (BOS ile)
    2. TextGenerator ile uret
    3. Yalnizca yeni tokenlari decode et

  ResetConversation()
    KV-cache sifirlar (yeni sohbet turu)


================================================================================
ChatCli komutlari
================================================================================

  /exit, /quit   Cikis
  /reset         Cache sifirla


================================================================================
KULLANIM
================================================================================

  CMakeProject2 demo
  CMakeProject2 chat --model model.ckptq --tokenizer tokenizer_data


================================================================================
IMPLEMENTASYON
================================================================================

  src/cli/chat_session.cpp
  src/cli/chat_cli.cpp

================================================================================
