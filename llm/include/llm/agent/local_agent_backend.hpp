#pragma once

#include <memory>
#include <mutex>
#include <random>
#include <string>

#include "llm/agent/agent_backend.hpp"
#include "llm/agent/tool.hpp"
#include "llm/cli/chat_session.hpp"

namespace llm::agent {

struct AgentConversationEntry {
  std::string role;
  std::string content;
};

class LocalAgentBackend final : public AgentBackend {
public:
  [[nodiscard]] Status Load(const AgentConfig& config) override;

  [[nodiscard]] AgentResponse Execute(const AgentRequest& request) override;

  [[nodiscard]] Status Reset() override;

  [[nodiscard]] ToolRegistry& GetToolRegistry() override;

  [[nodiscard]] const ToolRegistry& GetToolRegistry() const override;

  [[nodiscard]] bool IsLoaded() const override;

private:
  [[nodiscard]] std::string BuildAgentPrompt(
      const std::string& userMessage,
      const std::string& systemPrompt,
      const std::vector<AgentConversationEntry>& conversation) const;

  [[nodiscard]] std::optional<std::vector<ToolCall>> ParseToolCalls(
      const std::string& modelOutput) const;

  [[nodiscard]] std::string FormatToolResults(
      const std::vector<ToolResult>& results) const;

  [[nodiscard]] bool LooksLikeFinalAnswer(const std::string& text) const;

  AgentConfig config_;
  ToolRegistry toolRegistry_;
  cli::ChatSession session_;
  std::vector<AgentConversationEntry> conversation_;
  std::mt19937 rng_{std::random_device{}()};
  mutable std::mutex mutex_;
  bool loaded_ = false;
};

namespace builtin_tools {

ToolInfo EchoToolInfo();

ToolResult EchoToolExecute(const std::map<std::string, std::string>& args);

ToolInfo CalculatorToolInfo();

ToolResult CalculatorToolExecute(const std::map<std::string, std::string>& args);

void RegisterBuiltinTools(ToolRegistry& registry);

} // namespace builtin_tools

} // namespace llm::agent
