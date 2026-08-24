#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/training/chat_corpus.hpp"

namespace llm::training {

[[nodiscard]] std::optional<std::string> ExtractJsonStringField(const std::string& json, const std::string& key,
                                                              std::size_t searchFrom = 0);

[[nodiscard]] std::vector<std::pair<std::string, std::string>> ExtractRoleContentPairs(const std::string& json);

[[nodiscard]] Result<DialogueSample> ParseJsonlDialogueLine(const std::string& line);

} // namespace llm::training
