#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "llm/core/status.hpp"

namespace llm::server {

struct ChatRequest {
  std::string prompt;
  std::string intent;
  std::size_t maxTokens = 64;
  bool greedy = false;
  float temperature = 0.8f;
  std::uint32_t seed = 0;
};

struct ChatResponse {
  std::string text;
  std::string intent;
  std::string error;
  bool allowAutoLearn = false;
};

struct BackendInfo {
  std::string backend;
  std::string model;
  std::string stage;
  bool loaded = false;
};

class ChatBackend {
public:
  virtual ~ChatBackend() = default;

  [[nodiscard]] virtual ChatResponse Complete(const ChatRequest& request) = 0;

  [[nodiscard]] virtual Status Reset() = 0;

  [[nodiscard]] virtual BackendInfo GetInfo() const = 0;
};

} // namespace llm::server
