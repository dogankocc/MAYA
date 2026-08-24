#include "llm/nlp/corpus_retriever.hpp"

#include "llm/nlp/intent_classifier.hpp"

#include <algorithm>
#include <cctype>

namespace llm::nlp {

namespace {

std::string NormalizePrompt(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  value.erase(std::remove_if(value.begin(), value.end(),
                             [](const unsigned char ch) {
                               return std::ispunct(ch) != 0 && ch != '?' && ch != '\'';
                             }),
              value.end());
  const auto isSpace = [](const unsigned char ch) { return std::isspace(ch) != 0; };
  while (!value.empty() && isSpace(value.front())) {
    value.erase(value.begin());
  }
  while (!value.empty() && isSpace(value.back())) {
    value.pop_back();
  }
  return value;
}

std::vector<std::string> PromptTokens(const std::string& text) {
  std::vector<std::string> tokens;
  std::string current;
  for (const unsigned char ch : text) {
    if (std::isalnum(ch) != 0) {
      current.push_back(static_cast<char>(ch));
    } else if (!current.empty()) {
      if (current.size() >= 2) {
        tokens.push_back(current);
      }
      current.clear();
    }
  }
  if (!current.empty() && current.size() >= 2) {
    tokens.push_back(current);
  }
  return tokens;
}

} // namespace

void CorpusRetriever::SetSamples(std::vector<training::DialogueSample> samples) {
  samples_ = std::move(samples);
}

void CorpusRetriever::LoadDefaultKnowledgeBase() {
  std::vector<training::DialogueSample> merged = training::DefaultDialogueCorpus();

  const auto mergeFile = [&merged](const std::string& path) {
    const auto loaded = training::LoadDialogueCorpus(path);
    if (loaded.IsOk()) {
      merged = training::MergeDialogueCorpora(std::move(merged), loaded.Value());
    }
  };

  mergeFile("data/corpus/profile_dogan.jsonl");
  mergeFile("data/corpus/auto_learn.jsonl");
  samples_ = std::move(merged);
}

int CorpusRetriever::ScorePromptPair(const std::string& query, const std::string& samplePrompt) {
  const std::string normalizedQuery = NormalizePrompt(query);
  const std::string normalizedSample = NormalizePrompt(samplePrompt);
  if (normalizedQuery.empty() || normalizedSample.empty()) {
    return 0;
  }

  if (normalizedQuery == normalizedSample) {
    return 1000;
  }

  int score = 0;
  if (normalizedQuery.find(normalizedSample) != std::string::npos ||
      normalizedSample.find(normalizedQuery) != std::string::npos) {
    score += 120;
  }

  const auto queryTokens = PromptTokens(normalizedQuery);
  const auto sampleTokens = PromptTokens(normalizedSample);
  for (const std::string& queryToken : queryTokens) {
    for (const std::string& sampleToken : sampleTokens) {
      if (queryToken == sampleToken) {
        score += 35;
      } else if (queryToken.size() >= 4 && sampleToken.find(queryToken) != std::string::npos) {
        score += 20;
      } else if (sampleToken.size() >= 4 && queryToken.find(sampleToken) != std::string::npos) {
        score += 20;
      }
    }
  }

  return score;
}

std::optional<CorpusMatch> CorpusRetriever::FindBestMatch(const std::string& userPrompt,
                                                          const std::string& intent) const {
  if (userPrompt.empty() || samples_.empty()) {
    return std::nullopt;
  }

  const std::string normalizedIntent = NormalizeIntentLabel(intent);
  int bestScore = 0;
  const training::DialogueSample* bestSample = nullptr;

  for (const training::DialogueSample& sample : samples_) {
    int score = ScorePromptPair(userPrompt, sample.prompt);
    if (!sample.intent.empty() && NormalizeIntentLabel(sample.intent) == normalizedIntent) {
      score += 15;
    }
    if (score > bestScore) {
      bestScore = score;
      bestSample = &sample;
    }
  }

  constexpr int kMinScore = 35;
  if (bestSample == nullptr || bestScore < kMinScore) {
    return std::nullopt;
  }

  return CorpusMatch{
      .intent = bestSample->intent.empty() ? normalizedIntent : NormalizeIntentLabel(bestSample->intent),
      .response = bestSample->response,
      .score = bestScore,
  };
}

} // namespace llm::nlp
