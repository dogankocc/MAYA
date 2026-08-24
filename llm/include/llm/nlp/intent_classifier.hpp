#pragma once

#include <string>

namespace llm::nlp {

[[nodiscard]] std::string NormalizeIntentLabel(std::string value);

[[nodiscard]] std::string ClassifyIntent(const std::string& userPrompt, const std::string& assistantResponse = "");

[[nodiscard]] bool IsKnownIntent(const std::string& intent);

[[nodiscard]] std::string DefaultIntent();

} // namespace llm::nlp
