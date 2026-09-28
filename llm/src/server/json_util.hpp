#pragma once

#include <optional>
#include <string>
#include <vector>

#include "llm/server/chat_backend.hpp"
#include "llm/server/auto_learn_service.hpp"
#include "llm/server/hybrid_chat_backend.hpp"

namespace llm::server::detail {

[[nodiscard]] std::optional<ChatRequest> ParseChatRequest(const std::string& body);

[[nodiscard]] std::optional<std::string> ExtractStringField(const std::string& body, const std::string& key);

[[nodiscard]] std::optional<bool> ExtractBoolField(const std::string& body, const std::string& key);

[[nodiscard]] std::optional<float> ExtractNumberField(const std::string& body, const std::string& key);

[[nodiscard]] std::optional<std::size_t> ExtractSizeField(const std::string& body, const std::string& key);

[[nodiscard]] std::optional<std::vector<std::string>> ExtractStringArrayField(const std::string& body,
                                                                              const std::string& key);

[[nodiscard]] std::string JsonEscape(const std::string& value);

[[nodiscard]] std::string JsonStringArray(const std::vector<std::string>& values);

[[nodiscard]] std::string BuildChatResponseJson(const ChatResponse& response);

[[nodiscard]] std::string BuildHealthJson(const BackendInfo& info, const AutoLearnStatus& autoLearn = {});

[[nodiscard]] std::string BuildConfigJson(const ServerConfig& config);

[[nodiscard]] std::optional<ServerConfig> ParseConfigRequest(const std::string& body);

[[nodiscard]] std::string BuildOkJson();

} // namespace llm::server::detail
