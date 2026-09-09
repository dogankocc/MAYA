#include "llm/cli/agent_cli.hpp"

#include <iomanip>
#include <sstream>
#include <string>

#include "llm/core/logger.hpp"

namespace llm::cli {

AgentCli::AgentCli(agent::AgentBackend& backend) : backend_(backend) {}

void AgentCli::PrintWelcome(std::ostream& output) const {
  output << "╔════════════════════════════════════════════════════════════╗\n";
  output << "║           MAYA Agent Mode                                    ║\n";
  output << "╠════════════════════════════════════════════════════════════╣\n";
  output << "║ Commands:                                                    ║\n";
  output << "║   /help     - Show this help message                        ║\n";
  output << "║   /tools    - List available tools                          ║\n";
  output << "║   /reset    - Reset agent conversation                      ║\n";
  output << "║   /exit     - Quit (or /quit)                               ║\n";
  output << "╚════════════════════════════════════════════════════════════╝\n\n";
}

void AgentCli::PrintHelp(std::ostream& output) const {
  output << "\nAgent Mode Help:\n";
  output << "  /help     - Show this message\n";
  output << "  /tools    - List all available tools\n";
  output << "  /reset    - Clear conversation history\n";
  output << "  /exit     - Exit the program\n\n";
  output << "The agent can use tools to accomplish tasks. Simply ask a question\n";
  output << "and the agent will decide whether to use tools or answer directly.\n\n";
}

void AgentCli::PrintStep(const agent::AgentStep& step, std::ostream& output) const {
  output << "\n  [Step " << step.stepNumber << "] ";

  if (step.thought.action == "tool_call") {
    output << "Tool Call\n";
    for (const auto& call : step.thought.toolCalls) {
      output << "    → " << call.toolName << "(";
      bool first = true;
      for (const auto& [k, v] : call.arguments) {
        if (!first) {
          output << ", ";
        }
        output << k << "=\"" << v << "\"";
        first = false;
      }
      output << ")\n";
    }

    for (const auto& result : step.toolResults) {
      if (result.success) {
        output << "    ✓ Result: " << result.content << "\n";
      } else {
        output << "    ✗ Error: " << result.error << "\n";
      }
    }
  } else {
    output << "Reasoning: " << step.thought.reasoning.substr(0, 80);
    if (step.thought.reasoning.size() > 80) {
      output << "...";
    }
    output << "\n";
  }
}

bool AgentCli::ExecuteCommand(const std::string& line, std::ostream& output) {
  if (line == "/help" || line == "/?") {
    PrintHelp(output);
    return true;
  }

  if (line == "/tools") {
    output << "\nAvailable Tools:\n";
    const auto tools = backend_.GetToolRegistry().ListAllTools();
    if (tools.empty()) {
      output << "  (No tools registered)\n";
    } else {
      for (const auto& info : tools) {
        output << "  • " << info.name << ": " << info.description << "\n";
        if (!info.parameters.empty()) {
          output << "    Parameters: ";
          bool first = true;
          for (const auto& p : info.parameters) {
            if (!first) {
              output << ", ";
            }
            output << p.name << " (" << p.type << ")";
            if (p.required) {
              output << " [required]";
            }
            first = false;
          }
          output << "\n";
        }
      }
    }
    output << "\n";
    return true;
  }

  if (line == "/reset") {
    const Status status = backend_.Reset();
    if (status.IsOk()) {
      output << "Agent conversation reset.\n\n";
    } else {
      output << "Reset failed: " << status.Message() << "\n\n";
    }
    return true;
  }

  if (line == "/exit" || line == "/quit") {
    running_ = false;
    return true;
  }

  return false;
}

int AgentCli::Run(std::istream& input, std::ostream& output) {
  if (!backend_.IsLoaded()) {
    output << "Agent is not loaded. Use Load() first.\n";
    return 1;
  }

  PrintWelcome(output);
  running_ = true;

  std::string line;
  while (running_) {
    output << "\nAgent> ";
    output.flush();

    if (!std::getline(input, line)) {
      break;
    }

    if (line.empty()) {
      continue;
    }

    if (line[0] == '/') {
      ExecuteCommand(line, output);
      continue;
    }

    agent::AgentRequest request;
    request.userMessage = line;
    request.maxSteps = 10;
    request.maxTokensPerStep = 512;
    request.temperature = 0.7f;

    output << "  Thinking...\n";

    const agent::AgentResponse response = backend_.Execute(request);

    for (const auto& step : response.steps) {
      PrintStep(step, output);
    }

    output << "\n";
    if (response.success) {
      output << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
      output << response.finalAnswer << "\n";
      output << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

      Logger::Instance().Info("agent",
          "request completed: steps=" + std::to_string(response.totalSteps));
    } else {
      output << "❌ Error: " << response.error << "\n";
      Logger::Instance().Error("agent", "request failed: " + response.error);
    }
  }

  output << "\nBye.\n";
  return 0;
}

} // namespace llm::cli
