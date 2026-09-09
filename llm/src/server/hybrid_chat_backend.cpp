#include "llm/server/hybrid_chat_backend.hpp"

namespace llm::server {

HybridChatBackend::HybridChatBackend(HybridBackendConfig config) : config_(std::move(config)) {
  local_ = std::make_unique<LocalChatBackend>();
  localReady_ = local_->Load(config_.modelPath, config_.tokenizerPath).IsOk();

  OpenAiBackendConfig openAiConfig;
  openAiConfig.host = config_.apiHost;
  openAiConfig.apiKey = config_.apiKey;
  openAiConfig.model = config_.openAiModel;
  openAi_ = std::make_unique<OpenAiChatBackend>(std::move(openAiConfig));

  if (config_.enableAgentMode && localReady_) {
    agent_ = std::make_unique<agent::LocalAgentBackend>();
    agent::AgentConfig agentConfig;
    agentConfig.modelPath = config_.modelPath;
    agentConfig.tokenizerPath = config_.tokenizerPath;
    agentReady_ = agent_->Load(agentConfig).IsOk();
  }

  if (config_.activeBackend == "local" && !localReady_) {
    config_.activeBackend = "openai";
  }
}

ChatBackend* HybridChatBackend::ActiveBackend() {
  if (config_.activeBackend == "local" && localReady_) {
    return local_.get();
  }
  return openAi_.get();
}

ChatBackend* HybridChatBackend::ActiveBackend() const {
  if (config_.activeBackend == "local" && localReady_) {
    return local_.get();
  }
  return openAi_.get();
}

ChatResponse HybridChatBackend::Complete(const ChatRequest& request) {
  std::lock_guard<std::mutex> lock(mutex_);
  return ActiveBackend()->Complete(request);
}

Status HybridChatBackend::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (localReady_) {
    const Status localReset = local_->Reset();
    if (!localReset.IsOk()) {
      return localReset;
    }
  }
  return openAi_->Reset();
}

BackendInfo HybridChatBackend::GetInfo() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ActiveBackend()->GetInfo();
}

ServerConfig HybridChatBackend::GetConfig() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ServerConfig{
      .backend = config_.activeBackend,
      .model = ActiveBackend()->GetInfo().model,
      .localAvailable = localReady_,
      .openAiAvailable = true,
      .openAiModel = config_.openAiModel,
  };
}

Status HybridChatBackend::SetActiveBackend(const std::string& backend) {
  if (backend != "local" && backend != "openai") {
    return Status::Fail(ErrorCode::InvalidArgument, "backend must be local or openai");
  }
  if (backend == "local" && !localReady_) {
    return Status::Fail(ErrorCode::InvalidArgument, "local model is not available");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  config_.activeBackend = backend;
  return Status::Ok();
}

Status HybridChatBackend::SetOpenAiModel(const std::string& model) {
  if (model.empty()) {
    return Status::Fail(ErrorCode::InvalidArgument, "openai model name is empty");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  config_.openAiModel = model;

  OpenAiBackendConfig openAiConfig;
  openAiConfig.host = config_.apiHost;
  openAiConfig.apiKey = config_.apiKey;
  openAiConfig.model = model;
  openAi_ = std::make_unique<OpenAiChatBackend>(std::move(openAiConfig));
  return Status::Ok();
}

Status HybridChatBackend::ReloadLocalModel() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!local_) {
    local_ = std::make_unique<LocalChatBackend>();
  }
  localReady_ = local_->Load(config_.modelPath, config_.tokenizerPath).IsOk();
  if (!localReady_) {
    return Status::Fail(ErrorCode::Internal, "failed to reload local model");
  }

  if (config_.enableAgentMode) {
    if (!agent_) {
      agent_ = std::make_unique<agent::LocalAgentBackend>();
    }
    agent::AgentConfig agentConfig;
    agentConfig.modelPath = config_.modelPath;
    agentConfig.tokenizerPath = config_.tokenizerPath;
    agentReady_ = agent_->Load(agentConfig).IsOk();
  }

  return Status::Ok();
}

agent::AgentResponse HybridChatBackend::ExecuteAgent(const agent::AgentRequest& request) {
  agent::AgentResponse response;
  response.success = false;

  std::lock_guard<std::mutex> lock(mutex_);
  if (!agentReady_ || !agent_) {
    response.error = "agent mode not available";
    return response;
  }

  return agent_->Execute(request);
}

agent::ToolRegistry& HybridChatBackend::GetAgentToolRegistry() {
  if (!agent_) {
    agent_ = std::make_unique<agent::LocalAgentBackend>();
  }
  return agent_->GetToolRegistry();
}

bool HybridChatBackend::IsAgentModeAvailable() const {
  return agentReady_;
}

} // namespace llm::server
