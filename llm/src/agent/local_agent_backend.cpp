#include "llm/agent/local_agent_backend.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <sstream>

#include "llm/core/logger.hpp"

namespace llm::agent {

namespace {

constexpr const char* kToolCallStartMarker = "[TOOL_CALLS]";
constexpr const char* kToolCallEndMarker = "[/TOOL_CALLS]";
constexpr const char* kFinalAnswerMarker = "[FINAL_ANSWER]";

bool ContainsSubstring(const std::string& text, const std::string& substr) {
  return text.find(substr) != std::string::npos;
}

std::string TrimWhitespace(const std::string& s) {
  std::string result = s;
  result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](int ch) {
    return !std::isspace(ch);
  }));
  result.erase(std::find_if(result.rbegin(), result.rend(), [](int ch) {
    return !std::isspace(ch);
  }).base(), result.end());
  return result;
}

std::size_t FindMatchingBracket(const std::string& s, std::size_t start) {
  int depth = 0;
  bool inString = false;
  bool escaped = false;

  for (std::size_t i = start; i < s.size(); ++i) {
    const char ch = s[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch == '\\') {
      escaped = true;
      continue;
    }
    if (ch == '"') {
      inString = !inString;
      continue;
    }
    if (!inString) {
      if (ch == '{') {
        depth++;
      } else if (ch == '}') {
        depth--;
        if (depth == 0) {
          return i;
        }
      }
    }
  }
  return std::string::npos;
}

std::optional<std::string> ExtractJsonValue(const std::string& json, const std::string& key) {
  const std::string pattern = "\"" + key + "\"";
  std::size_t pos = 0;

  while ((pos = json.find(pattern, pos)) != std::string::npos) {
    pos += pattern.size();
    while (pos < json.size() && std::isspace(json[pos])) {
      pos++;
    }
    if (pos >= json.size() || json[pos] != ':') {
      continue;
    }
    pos++;
    while (pos < json.size() && std::isspace(json[pos])) {
      pos++;
    }
    if (pos >= json.size() || json[pos] != '"') {
      continue;
    }
    pos++;
    std::size_t start = pos;
    bool escaped = false;
    while (pos < json.size()) {
      if (escaped) {
        escaped = false;
        pos++;
        continue;
      }
      if (json[pos] == '\\') {
        escaped = true;
        pos++;
        continue;
      }
      if (json[pos] == '"') {
        return json.substr(start, pos - start);
      }
      pos++;
    }
  }
  return std::nullopt;
}

std::optional<std::map<std::string, std::string>> ExtractJsonObject(const std::string& json) {
  std::map<std::string, std::string> result;

  std::size_t pos = 0;
  while (pos < json.size()) {
    while (pos < json.size() && (std::isspace(json[pos]) || json[pos] == '{' || json[pos] == '}')) {
      pos++;
    }
    if (pos >= json.size()) {
      break;
    }
    if (json[pos] != '"') {
      pos++;
      continue;
    }
    pos++;
    std::size_t keyStart = pos;
    while (pos < json.size() && json[pos] != '"') {
      if (json[pos] == '\\') {
        pos += 2;
      } else {
        pos++;
      }
    }
    const std::string key = json.substr(keyStart, pos - keyStart);
    pos++;

    while (pos < json.size() && (std::isspace(json[pos]) || json[pos] == ':')) {
      pos++;
    }
    if (pos >= json.size()) {
      break;
    }

    if (json[pos] == '"') {
      pos++;
      std::size_t valStart = pos;
      while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\') {
          pos += 2;
        } else {
          pos++;
        }
      }
      result[key] = json.substr(valStart, pos - valStart);
      pos++;
    } else if (json[pos] == '{') {
      const std::size_t end = FindMatchingBracket(json, pos);
      if (end != std::string::npos) {
        result[key] = json.substr(pos, end - pos + 1);
        pos = end + 1;
      } else {
        pos++;
      }
    } else {
      std::size_t valStart = pos;
      while (pos < json.size() && json[pos] != ',' && json[pos] != '}') {
        pos++;
      }
      result[key] = TrimWhitespace(json.substr(valStart, pos - valStart));
    }
  }

  return result;
}

} // namespace

Status LocalAgentBackend::Load(const AgentConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  config_ = config;

  const Status status = session_.Load(config.modelPath, config.tokenizerPath);
  if (!status.IsOk()) {
    Logger::Instance().Error("agent", "failed to load model: " + status.Message());
    return status;
  }

  session_.Sampling().temperature = config.temperature;

  if (config.enableTools) {
    builtin_tools::RegisterBuiltinTools(toolRegistry_);
  }

  conversation_.clear();
  loaded_ = true;

  Logger::Instance().Info("agent", "local agent backend loaded");
  return Status::Ok();
}

AgentResponse LocalAgentBackend::Execute(const AgentRequest& request) {
  AgentResponse response;
  response.success = false;

  std::lock_guard<std::mutex> lock(mutex_);

  if (!loaded_) {
    response.error = "agent not loaded";
    return response;
  }

  session_.ResetConversation();
  conversation_.clear();

  const std::size_t maxSteps = std::min(request.maxSteps, config_.maxSteps);
  std::vector<AgentStep> steps;

  for (std::size_t step = 0; step < maxSteps; ++step) {
    AgentStep currentStep;
    currentStep.stepNumber = step + 1;

    const std::string prompt = BuildAgentPrompt(
        request.userMessage,
        request.systemPrompt.empty() ? config_.systemPrompt : request.systemPrompt,
        conversation_);

    inference::SamplingConfig& sampling = session_.Sampling();
    const float savedTemp = sampling.temperature;
    sampling.temperature = request.temperature;

    const auto result = session_.Complete(prompt, request.maxTokensPerStep, rng_);
    sampling.temperature = savedTemp;

    if (!result.IsOk()) {
      Logger::Instance().Error("agent", "step " + std::to_string(step + 1) + " failed: " + result.GetError().message);
      currentStep.observation = "Generation failed: " + result.GetError().message;
      steps.push_back(std::move(currentStep));
      break;
    }

    const std::string modelOutput = result.Value();
    Logger::Instance().Debug("agent", "step " + std::to_string(step + 1) + " output: " + modelOutput.substr(0, 200));

    const auto toolCalls = ParseToolCalls(modelOutput);

    if (toolCalls && !toolCalls->empty() && config_.enableTools) {
      currentStep.thought.action = "tool_call";
      currentStep.thought.toolCalls = *toolCalls;

      std::vector<ToolResult> toolResults;
      for (const ToolCall& call : *toolCalls) {
        const ToolResult toolResult = toolRegistry_.Execute(call.toolName, call.arguments);
        toolResults.push_back(toolResult);

        Logger::Instance().Info("agent",
            "tool: " + call.toolName + " -> " + (toolResult.success ? "success" : "failed: " + toolResult.error));
      }

      currentStep.toolResults = toolResults;
      currentStep.observation = FormatToolResults(toolResults);

      conversation_.push_back({"assistant", modelOutput});
      conversation_.push_back({"tool", currentStep.observation});
    } else {
      currentStep.thought.action = "final_answer";
      currentStep.thought.reasoning = modelOutput;

      if (LooksLikeFinalAnswer(modelOutput)) {
        const std::size_t markerPos = modelOutput.find(kFinalAnswerMarker);
        if (markerPos != std::string::npos) {
          response.finalAnswer = TrimWhitespace(modelOutput.substr(markerPos + std::string(kFinalAnswerMarker).size()));
        } else {
          response.finalAnswer = modelOutput;
        }
      } else {
        response.finalAnswer = modelOutput;
      }

      conversation_.push_back({"assistant", modelOutput});
      steps.push_back(std::move(currentStep));
      response.success = true;
      break;
    }

    steps.push_back(std::move(currentStep));
  }

  response.steps = std::move(steps);
  response.totalSteps = response.steps.size();

  Logger::Instance().Info("agent",
      "execution complete: steps=" + std::to_string(response.totalSteps) +
      " success=" + (response.success ? "yes" : "no"));

  return response;
}

Status LocalAgentBackend::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  session_.ResetConversation();
  conversation_.clear();
  return Status::Ok();
}

ToolRegistry& LocalAgentBackend::GetToolRegistry() {
  return toolRegistry_;
}

const ToolRegistry& LocalAgentBackend::GetToolRegistry() const {
  return toolRegistry_;
}

bool LocalAgentBackend::IsLoaded() const {
  return loaded_;
}

std::string LocalAgentBackend::BuildAgentPrompt(
    const std::string& userMessage,
    const std::string& systemPrompt,
    const std::vector<AgentConversationEntry>& conversation) const {
  std::ostringstream oss;

  oss << systemPrompt << "\n\n";

  if (config_.enableTools && !toolRegistry_.ListAllTools().empty()) {
    oss << "AVAILABLE TOOLS:\n";
    for (const ToolInfo& info : toolRegistry_.ListAllTools()) {
      oss << "- " << info.name << ": " << info.description << "\n";
      oss << "  Parameters: ";
      bool first = true;
      for (const ToolParameter& p : info.parameters) {
        if (!first) {
          oss << ", ";
        }
        oss << p.name << " (" << p.type << ")";
        if (p.required) {
          oss << " [required]";
        }
        first = false;
      }
      oss << "\n";
    }
    oss << "\n";
    oss << "To call tools, use this EXACT format:\n";
    oss << kToolCallStartMarker << "\n";
    oss << "[{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}]\n";
    oss << kToolCallEndMarker << "\n\n";
    oss << "When you have the final answer, use: " << kFinalAnswerMarker << " <your answer>\n\n";
  }

  for (const AgentConversationEntry& entry : conversation) {
    oss << entry.role << ": " << entry.content << "\n";
  }

  oss << "user: " << userMessage << "\n";
  oss << "assistant: ";

  return oss.str();
}

std::optional<std::vector<ToolCall>> LocalAgentBackend::ParseToolCalls(
    const std::string& modelOutput) const {
  const std::size_t startPos = modelOutput.find(kToolCallStartMarker);
  if (startPos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t endPos = modelOutput.find(kToolCallEndMarker, startPos);
  if (endPos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t jsonStart = startPos + std::string(kToolCallStartMarker).size();
  const std::string jsonContent = TrimWhitespace(modelOutput.substr(jsonStart, endPos - jsonStart));

  Logger::Instance().Debug("agent", "parsing tool calls from: " + jsonContent);

  std::vector<ToolCall> result;

  std::size_t pos = 0;
  while (pos < jsonContent.size()) {
    while (pos < jsonContent.size() && (std::isspace(jsonContent[pos]) || jsonContent[pos] == '[' || jsonContent[pos] == ',')) {
      pos++;
    }
    if (pos >= jsonContent.size() || jsonContent[pos] != '{') {
      break;
    }

    const std::size_t objEnd = FindMatchingBracket(jsonContent, pos);
    if (objEnd == std::string::npos) {
      break;
    }

    const std::string objStr = jsonContent.substr(pos, objEnd - pos + 1);
    pos = objEnd + 1;

    const auto name = ExtractJsonValue(objStr, "name");
    if (!name) {
      Logger::Instance().Warn("agent", "tool call missing 'name' field");
      continue;
    }

    ToolCall call;
    call.toolName = *name;

    const auto argsObj = ExtractJsonObject(objStr);
    if (argsObj) {
      const auto argsIt = argsObj->find("arguments");
      if (argsIt != argsObj->end()) {
        const auto nestedArgs = ExtractJsonObject(argsIt->second);
        if (nestedArgs) {
          call.arguments = *nestedArgs;
        }
      }
    }

    result.push_back(std::move(call));
  }

  return result.empty() ? std::nullopt : std::make_optional(result);
}

std::string LocalAgentBackend::FormatToolResults(const std::vector<ToolResult>& results) const {
  std::ostringstream oss;
  oss << "Tool Results:\n";
  for (const ToolResult& r : results) {
    oss << "[" << r.toolName << "] ";
    if (r.success) {
      oss << "Success: " << r.content;
    } else {
      oss << "Error: " << r.error;
    }
    oss << "\n";
  }
  return oss.str();
}

bool LocalAgentBackend::LooksLikeFinalAnswer(const std::string& text) const {
  return ContainsSubstring(text, kFinalAnswerMarker) ||
         !ContainsSubstring(text, kToolCallStartMarker);
}

namespace builtin_tools {

ToolInfo EchoToolInfo() {
  ToolInfo info;
  info.name = "echo";
  info.description = "Returns the input message. Useful for testing.";
  info.parameters = {
    ToolParameter{.name = "message", .type = "string", .description = "The message to echo", .required = true}
  };
  return info;
}

ToolResult EchoToolExecute(const std::map<std::string, std::string>& args) {
  ToolResult result;
  result.toolName = "echo";
  result.success = true;

  const auto it = args.find("message");
  if (it == args.end()) {
    result.success = false;
    result.error = "missing 'message' parameter";
    return result;
  }

  result.content = "Echo: " + it->second;
  return result;
}

ToolInfo CalculatorToolInfo() {
  ToolInfo info;
  info.name = "calculator";
  info.description = "Performs basic arithmetic calculations. Use for math operations.";
  info.parameters = {
    ToolParameter{.name = "expression", .type = "string", .description = "The math expression (e.g., '2 + 2', '10 * 5')", .required = true}
  };
  return info;
}

ToolResult CalculatorToolExecute(const std::map<std::string, std::string>& args) {
  ToolResult result;
  result.toolName = "calculator";

  const auto it = args.find("expression");
  if (it == args.end()) {
    result.success = false;
    result.error = "missing 'expression' parameter";
    return result;
  }

  const std::string expr = TrimWhitespace(it->second);

  std::size_t opPos = std::string::npos;
  char op = 0;
  for (std::size_t i = 0; i < expr.size(); ++i) {
    if (expr[i] == '+' || expr[i] == '-' || expr[i] == '*' || expr[i] == '/') {
      op = expr[i];
      opPos = i;
      break;
    }
  }

  if (opPos == std::string::npos) {
    result.success = false;
    result.error = "unsupported expression format. Use: number operator number";
    return result;
  }

  const std::string leftStr = TrimWhitespace(expr.substr(0, opPos));
  const std::string rightStr = TrimWhitespace(expr.substr(opPos + 1));

  double left = 0.0, right = 0.0;
  {
    auto [ptr, ec] = std::from_chars(leftStr.data(), leftStr.data() + leftStr.size(), left);
    if (ec != std::errc{}) {
      result.success = false;
      result.error = "invalid number: " + leftStr;
      return result;
    }
  }
  {
    auto [ptr, ec] = std::from_chars(rightStr.data(), rightStr.data() + rightStr.size(), right);
    if (ec != std::errc{}) {
      result.success = false;
      result.error = "invalid number: " + rightStr;
      return result;
    }
  }

  double output = 0.0;
  switch (op) {
  case '+': output = left + right; break;
  case '-': output = left - right; break;
  case '*': output = left * right; break;
  case '/':
    if (right == 0.0) {
      result.success = false;
      result.error = "division by zero";
      return result;
    }
    output = left / right;
    break;
  default:
    result.success = false;
    result.error = "unknown operator";
    return result;
  }

  result.success = true;
  result.content = std::to_string(output);
  return result;
}

void RegisterBuiltinTools(ToolRegistry& registry) {
  registry.RegisterFunction(EchoToolInfo(), EchoToolExecute);
  registry.RegisterFunction(CalculatorToolInfo(), CalculatorToolExecute);
}

} // namespace builtin_tools

} // namespace llm::agent
