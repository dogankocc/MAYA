#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "llm/agent/agent_backend.hpp"
#include "llm/agent/local_agent_backend.hpp"

namespace llm::cli {

class AgentCli {
public:
  explicit AgentCli(agent::AgentBackend& backend);

  int Run(std::istream& input, std::ostream& output);

  int Run() {
    return Run(std::cin, std::cout);
  }

private:
  void PrintWelcome(std::ostream& output) const;

  void PrintHelp(std::ostream& output) const;

  void PrintStep(const agent::AgentStep& step, std::ostream& output) const;

  bool ExecuteCommand(const std::string& line, std::ostream& output);

  agent::AgentBackend& backend_;
  bool running_ = false;
};

} // namespace llm::cli
