#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "llm/core/status.hpp"

namespace llm::agent {

struct ToolParameter {
  std::string name;
  std::string type;
  std::string description;
  bool required = false;
  std::string defaultValue;
};

struct ToolInfo {
  std::string name;
  std::string description;
  std::vector<ToolParameter> parameters;
};

struct ToolCall {
  std::string toolName;
  std::map<std::string, std::string> arguments;
};

struct ToolResult {
  std::string toolName;
  std::string content;
  bool success = true;
  std::string error;
};

using ToolFunction = std::function<ToolResult(const std::map<std::string, std::string>&)>;

class Tool {
public:
  virtual ~Tool() = default;

  [[nodiscard]] virtual ToolInfo GetInfo() const = 0;

  [[nodiscard]] virtual ToolResult Execute(const std::map<std::string, std::string>& args) = 0;
};

class FunctionTool final : public Tool {
public:
  FunctionTool(ToolInfo info, ToolFunction func);

  [[nodiscard]] ToolInfo GetInfo() const override;

  [[nodiscard]] ToolResult Execute(const std::map<std::string, std::string>& args) override;

private:
  ToolInfo info_;
  ToolFunction func_;
};

class ToolRegistry {
public:
  void RegisterTool(std::unique_ptr<Tool> tool);

  void RegisterFunction(const ToolInfo& info, ToolFunction func);

  [[nodiscard]] bool HasTool(const std::string& name) const;

  [[nodiscard]] Tool* GetTool(const std::string& name) const;

  [[nodiscard]] std::vector<ToolInfo> ListAllTools() const;

  [[nodiscard]] ToolResult Execute(const std::string& toolName, const std::map<std::string, std::string>& args);

private:
  std::map<std::string, std::unique_ptr<Tool>> tools_;
};

} // namespace llm::agent
