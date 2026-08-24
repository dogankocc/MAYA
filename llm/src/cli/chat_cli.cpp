#include "llm/cli/chat_cli.hpp"

#include <random>
#include <string>

namespace llm::cli {

ChatCli::ChatCli(ChatSession& session) : session_(session) {}

int ChatCli::Run(std::istream& input, std::ostream& output) {
  if (!session_.IsLoaded()) {
    output << "Chat session is not loaded. Use Load() first.\n";
    return 1;
  }

  output << "LLM Chat (commands: /exit, /reset)\n";

  std::mt19937 rng(static_cast<unsigned>(std::random_device{}()));
  std::string line;

  while (true) {
    output << "\n> ";
    output.flush();

    if (!std::getline(input, line)) {
      break;
    }

    if (line == "/exit" || line == "/quit") {
      break;
    }

    if (line == "/reset") {
      session_.ResetConversation();
      output << "Conversation reset.\n";
      continue;
    }

    if (line.empty()) {
      continue;
    }

    const auto reply = session_.Complete(line, 64, rng);
    if (!reply.IsOk()) {
      output << "[error] " << reply.GetError().message << '\n';
      continue;
    }

    output << reply.Value() << '\n';
  }

  output << "Bye.\n";
  return 0;
}

} // namespace llm::cli
