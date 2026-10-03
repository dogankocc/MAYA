#include "llm/training/dialogue_format.hpp"

#include "llm/nlp/intent_classifier.hpp"

#include <algorithm>
#include <cctype>

namespace {

std::string ToLowerAscii(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

bool ContainsKeyword(const std::string& haystack, const std::initializer_list<const char*> needles) {
  for (const char* needle : needles) {
    if (haystack.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool PromptMentionsUserCar(const std::string& userPrompt) {
  const std::string text = ToLowerAscii(userPrompt);
  return ContainsKeyword(text, {"aria", "araban", "araba", "arabam", "fiat", "punto", "multiAir", "otomobil"});
}

std::string NormalizeUserPrompt(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

} // namespace

namespace llm::training {

std::string ResolveIntent(const DialogueSample& sample) {
  if (!sample.intent.empty() && nlp::IsKnownIntent(sample.intent)) {
    return nlp::NormalizeIntentLabel(sample.intent);
  }
  return nlp::ClassifyIntent(sample.prompt, sample.response);
}

std::string BuildTrainingSequence(const DialogueSample& sample) {
  const std::string intent = ResolveIntent(sample);
  return BuildInferencePrompt(intent, sample.prompt, sample.system) + sample.response;
}

std::string BuildInferencePrompt(const std::string& intent, const std::string& userPrompt,
                                 const std::string& systemPrompt) {
  const std::string normalizedIntent =
      intent.empty() ? nlp::DefaultIntent() : nlp::NormalizeIntentLabel(intent);
  std::string sequence = "intent: " + normalizedIntent + ".";
  if (!systemPrompt.empty()) {
    sequence += " system: " + systemPrompt;
  }
  sequence += " user: " + NormalizeUserPrompt(userPrompt) + " assistant: ";
  return sequence;
}

std::string SanitizeGeneratedResponse(std::string text) {
  const std::string marker = "assistant:";
  const std::size_t markerPos = text.find(marker);
  if (markerPos != std::string::npos) {
    text.erase(0, markerPos + marker.size());
  }

  const std::string intentMarker = "intent:";
  const std::size_t intentPos = text.find(intentMarker);
  if (intentPos != std::string::npos) {
    text.erase(intentPos, text.size());
  }

  while (!text.empty() && (text.front() == ' ' || text.front() == '\n' || text.front() == '\r')) {
    text.erase(text.begin());
  }
  return text;
}

} // namespace llm::training
