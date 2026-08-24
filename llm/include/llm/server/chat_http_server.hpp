#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "llm/core/status.hpp"
#include "llm/server/auto_learn_service.hpp"
#include "llm/server/hybrid_chat_backend.hpp"

namespace llm::server {

class ChatHttpServer {
public:
  ChatHttpServer(std::unique_ptr<HybridChatBackend> backend, std::unique_ptr<AutoLearnService> autoLearn = nullptr);

  [[nodiscard]] Status Run(const std::string& host, int port);

  void RequestStop();

  [[nodiscard]] ChatResponse HandleChat(const ChatRequest& request);

  [[nodiscard]] Status HandleReset();

private:
  std::unique_ptr<HybridChatBackend> backend_;
  std::unique_ptr<AutoLearnService> autoLearn_;
  std::mutex mutex_;
  bool stopRequested_ = false;
};

} // namespace llm::server
