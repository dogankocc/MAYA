#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/core/types.hpp"
#include "llm/tokenizer/vocabulary.hpp"

namespace llm {

using MergePair = std::pair<std::string, std::string>;

class BpeTokenizer {
public:
  BpeTokenizer();

  [[nodiscard]] const Vocabulary& GetVocabulary() const { return vocabulary_; }

  [[nodiscard]] Vocabulary& GetVocabulary() { return vocabulary_; }

  [[nodiscard]] Status Train(const std::string& corpus, std::size_t targetVocabSize);

  [[nodiscard]] std::vector<TokenId> Encode(const std::string& text) const;

  [[nodiscard]] std::string Decode(const std::vector<TokenId>& ids) const;

  [[nodiscard]] std::vector<TokenId> EncodeWithSpecialTokens(const std::string& text, bool addBos, bool addEos) const;

  [[nodiscard]] Status Save(const std::string& directory) const;

  [[nodiscard]] static Result<BpeTokenizer> Load(const std::string& directory);

private:
  [[nodiscard]] std::vector<std::string> ApplyBpe(const std::vector<std::string>& symbols) const;

  [[nodiscard]] int FindBestMerge(const std::vector<std::string>& symbols) const;

  Vocabulary vocabulary_;
  std::vector<MergePair> merges_;
  std::unordered_map<std::string, int> mergeRank_;
};

} // namespace llm
