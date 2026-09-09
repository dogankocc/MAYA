#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "llm/agent/agent_backend.hpp"
#include "llm/agent/local_agent_backend.hpp"
#include "llm/core/status.hpp"
#include "llm/server/chat_backend.hpp"
#include "llm/server/local_chat_backend.hpp"
#include "llm/server/openai_chat_backend.hpp"

namespace llm::server {

struct HybridBackendConfig {
  std::string modelPath = "model.ckptq";
  std::string tokenizerPath = "tokenizer_data";
  std::string apiHost = "127.0.0.1:11434";
  std::string apiKey;
  std::string openAiModel = "llama3.2";
  std::string activeBackend = "openai";
  bool enableAgentMode = true;
};

struct ServerConfig {
  std::string backend;
  std::string model;
  bool localAvailable = false;
  bool openAiAvailable = false;
  std::string openAiModel;
};

class HybridChatBackend final : public ChatBackend {
public:
  explicit HybridChatBackend(HybridBackendConfig config);

  [[nodiscard]] ChatResponse Complete(const ChatRequest& request) override;

  [[nodiscard]] Status Reset() override;

  [[nodiscard]] BackendInfo GetInfo() const override;

  [[nodiscard]] ServerConfig GetConfig() const;

  [[nodiscard]] Status SetActiveBackend(const std::string& backend);

  [[nodiscard]] Status SetOpenAiModel(const std::string& model);

  [[nodiscard]] Status ReloadLocalModel();

  [[nodiscard]] const HybridBackendConfig& GetBackendConfig() const { return config_; }

  [[nodiscard]] agent::AgentResponse ExecuteAgent(const agent::AgentRequest& request);

  [[nodiscard]] agent::ToolRegistry& GetAgentToolRegistry();

  [[nodiscard]] bool IsAgentModeAvailable() const;

private:
  [[nodiscard]] ChatBackend* ActiveBackend();

  [[nodiscard]] ChatBackend* ActiveBackend() const;

  HybridBackendConfig config_;
  std::unique_ptr<LocalChatBackend> local_;
  bool localReady_ = false;
  std::unique_ptr<OpenAiChatBackend> openAi_;
  mutable std::mutex mutex_;

  std::unique_ptr<agent::LocalAgentBackend> agent_;
  bool agentReady_ = false;
};

} // namespace llm::server
