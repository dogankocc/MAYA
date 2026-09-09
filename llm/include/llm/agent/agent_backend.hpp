#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "llm/agent/tool.hpp"
#include "llm/core/status.hpp"

namespace llm::agent {

struct AgentThought {
  std::string reasoning;
  std::string action;
  std::vector<ToolCall> toolCalls;
};

struct AgentStep {
  std::size_t stepNumber;
  AgentThought thought;
  std::vector<ToolResult> toolResults;
  std::string observation;
};

struct AgentRequest {
  std::string userMessage;
  std::string systemPrompt;
  std::size_t maxSteps = 10;
  std::size_t maxTokensPerStep = 512;
  float temperature = 0.7f;
  std::uint32_t seed = 0;
  bool streaming = false;
};

struct AgentResponse {
  std::string finalAnswer;
  std::vector<AgentStep> steps;
  std::size_t totalSteps;
  std::size_t totalTokens;
  bool success = true;
  std::string error;
};

struct AgentConfig {
  std::string modelPath = "model.ckptq";
  std::string tokenizerPath = "tokenizer_data";
  std::string systemPrompt = R"(You are a helpful AI assistant with tool use capabilities.

When answering:
1. Think about what the user needs
2. If you need information, use available tools
3. Format tool calls as JSON: [{"name": "tool_name", "arguments": {"key": "value"}}]
4. After getting tool results, formulate your final answer
5. Always provide clear, helpful responses)";
  std::size_t maxSteps = 10;
  float temperature = 0.7f;
  bool enableTools = true;
};

class AgentBackend {
public:
  virtual ~AgentBackend() = default;

  [[nodiscard]] virtual Status Load(const AgentConfig& config) = 0;

  [[nodiscard]] virtual AgentResponse Execute(const AgentRequest& request) = 0;

  [[nodiscard]] virtual Status Reset() = 0;

  [[nodiscard]] virtual ToolRegistry& GetToolRegistry() = 0;

  [[nodiscard]] virtual const ToolRegistry& GetToolRegistry() const = 0;

  [[nodiscard]] virtual bool IsLoaded() const = 0;
};

} // namespace llm::agent
