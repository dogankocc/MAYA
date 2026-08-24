#pragma once

#include <string>

#include "llm/training/chat_corpus.hpp"

namespace llm::training {

[[nodiscard]] std::string ResolveIntent(const DialogueSample& sample);

[[nodiscard]] std::string BuildTrainingSequence(const DialogueSample& sample);

[[nodiscard]] std::string BuildInferencePrompt(const std::string& intent, const std::string& userPrompt,
                                               const std::string& systemPrompt = "");

[[nodiscard]] std::string SanitizeGeneratedResponse(std::string text);

[[nodiscard]] bool IsLowQualityResponse(const std::string& text);

[[nodiscard]] bool IsFallbackResponse(const std::string& text);

[[nodiscard]] std::string FallbackResponseForIntent(const std::string& intent, const std::string& userPrompt);

} // namespace llm::training
