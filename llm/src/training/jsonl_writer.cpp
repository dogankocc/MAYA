#include "llm/training/jsonl_writer.hpp"

namespace llm::training {

std::string JsonEscapeForJsonl(const std::string& text) {
  std::string escaped;
  escaped.reserve(text.size() + 8);
  for (const char ch : text) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(ch);
      break;
    }
  }
  return escaped;
}

std::string FormatDialogueJsonlLine(const std::string& intent, const std::string& userText,
                                    const std::string& assistantText) {
  return std::string("{\"intent\":\"") + JsonEscapeForJsonl(intent) + "\",\"messages\":[{\"role\":\"user\",\"content\":\"" +
         JsonEscapeForJsonl(userText) + "\"},{\"role\":\"assistant\",\"content\":\"" + JsonEscapeForJsonl(assistantText) +
         "\"}]}\n";
}

} // namespace llm::training
