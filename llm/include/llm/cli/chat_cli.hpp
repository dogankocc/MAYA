#pragma once

#include <iostream>

#include "llm/cli/chat_session.hpp"

namespace llm::cli {

class ChatCli {
public:
  explicit ChatCli(ChatSession& session);

  [[nodiscard]] int Run(std::istream& input = std::cin, std::ostream& output = std::cout);

private:
  ChatSession& session_;
};

} // namespace llm::cli
