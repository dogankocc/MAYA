#include "llm/agent/tool.hpp"

namespace llm::agent {

FunctionTool::FunctionTool(ToolInfo info, ToolFunction func)
    : info_(std::move(info)), func_(std::move(func)) {}

ToolInfo FunctionTool::GetInfo() const {
  return info_;
}

ToolResult FunctionTool::Execute(const std::map<std::string, std::string>& args) {
  if (!func_) {
    ToolResult result;
    result.toolName = info_.name;
    result.success = false;
    result.error = "tool function is null";
    return result;
  }
  return func_(args);
}

void ToolRegistry::RegisterTool(std::unique_ptr<Tool> tool) {
  if (!tool) {
    return;
  }
  const ToolInfo info = tool->GetInfo();
  tools_[info.name] = std::move(tool);
}

void ToolRegistry::RegisterFunction(const ToolInfo& info, ToolFunction func) {
  RegisterTool(std::make_unique<FunctionTool>(info, std::move(func)));
}

bool ToolRegistry::HasTool(const std::string& name) const {
  return tools_.find(name) != tools_.end();
}

Tool* ToolRegistry::GetTool(const std::string& name) const {
  const auto it = tools_.find(name);
  if (it == tools_.end()) {
    return nullptr;
  }
  return it->second.get();
}

std::vector<ToolInfo> ToolRegistry::ListAllTools() const {
  std::vector<ToolInfo> result;
  result.reserve(tools_.size());
  for (const auto& [name, tool] : tools_) {
    result.push_back(tool->GetInfo());
  }
  return result;
}

ToolResult ToolRegistry::Execute(const std::string& toolName, const std::map<std::string, std::string>& args) {
  Tool* tool = GetTool(toolName);
  if (!tool) {
    ToolResult result;
    result.toolName = toolName;
    result.success = false;
    result.error = "tool not found: " + toolName;
    return result;
  }
  return tool->Execute(args);
}

} // namespace llm::agent
