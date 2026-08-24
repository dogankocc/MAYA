#pragma once

#include <optional>
#include <string>
#include <vector>

#include "llm/training/chat_corpus.hpp"

namespace llm::nlp {

struct CorpusMatch {
  std::string intent;
  std::string response;
  int score = 0;
};

class CorpusRetriever {
public:
  void SetSamples(std::vector<training::DialogueSample> samples);

  void LoadDefaultKnowledgeBase();

  [[nodiscard]] std::optional<CorpusMatch> FindBestMatch(const std::string& userPrompt,
                                                         const std::string& intent) const;

private:
  [[nodiscard]] static int ScorePromptPair(const std::string& query, const std::string& samplePrompt);

  std::vector<training::DialogueSample> samples_;
};

} // namespace llm::nlp
