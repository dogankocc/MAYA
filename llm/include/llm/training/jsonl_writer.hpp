#pragma once

#include <string>

namespace llm::training {

[[nodiscard]] std::string JsonEscapeForJsonl(const std::string& text);

[[nodiscard]] std::string FormatDialogueJsonlLine(const std::string& intent, const std::string& userText,
                                                  const std::string& assistantText);

} // namespace llm::training
