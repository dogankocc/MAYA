#pragma once

#include <mutex>
#include <random>
#include <string>

#include "llm/cli/chat_session.hpp"
#include "llm/nlp/corpus_retriever.hpp"
#include "llm/server/chat_backend.hpp"

namespace llm::server {

class LocalChatBackend final : public ChatBackend {
public:
  [[nodiscard]] Status Load(const std::string& modelPath, const std::string& tokenizerPath);

  [[nodiscard]] ChatResponse Complete(const ChatRequest& request) override;

  [[nodiscard]] Status Reset() override;

  [[nodiscard]] BackendInfo GetInfo() const override;

private:
  static constexpr float kMinGenerationConfidence = 0.12f;

  mutable std::mutex mutex_;
  cli::ChatSession session_;
  nlp::CorpusRetriever retriever_;
  std::mt19937 rng_{std::random_device{}()};
};

} // namespace llm::server
