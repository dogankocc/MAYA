#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "llm/server/chat_backend.hpp"

namespace llm::server {

struct OpenAiBackendConfig {
  std::string host = "127.0.0.1:11434";
  std::string apiKey;
  std::string model = "llama3.2";
  std::string systemPrompt =
      "Sen yardimci, nazik ve net cevaplar veren bir Turkce yapay zeka asistanisin. "
      "Kullaniciya kisa ve anlasilir yanitlar ver.";
  long timeoutSeconds = 120;
};

class OpenAiChatBackend final : public ChatBackend {
public:
  explicit OpenAiChatBackend(OpenAiBackendConfig config);

  [[nodiscard]] ChatResponse Complete(const ChatRequest& request) override;

  [[nodiscard]] Status Reset() override;

  [[nodiscard]] BackendInfo GetInfo() const override;

private:
  struct Message {
    std::string role;
    std::string content;
  };

  [[nodiscard]] std::string BuildRequestBody(const ChatRequest& request) const;

  OpenAiBackendConfig config_;
  mutable std::mutex mutex_;
  std::vector<Message> messages_;
};

} // namespace llm::server
