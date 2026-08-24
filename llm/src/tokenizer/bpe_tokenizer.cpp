#include "llm/tokenizer/bpe_tokenizer.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace llm {

namespace {

using Word = std::vector<std::string>;

std::string MergeKey(const std::string& left, const std::string& right) {
  return left + "\x1F" + right;
}

std::vector<Word> SplitCorpusIntoWords(const std::string& corpus) {
  std::vector<Word> words;
  std::istringstream stream(corpus);
  std::string token;

  while (stream >> token) {
    Word word;
    word.push_back(kWordStartToken);
    for (const char ch : token) {
      word.emplace_back(1, ch);
    }
    if (word.size() > 1) {
      words.push_back(std::move(word));
    }
  }

  return words;
}

std::unordered_map<std::string, std::size_t> CountPairFrequencies(const std::vector<Word>& words) {
  std::unordered_map<std::string, std::size_t> frequencies;

  for (const Word& word : words) {
    if (word.size() < 2) {
      continue;
    }

    for (std::size_t i = 0; i + 1 < word.size(); ++i) {
      ++frequencies[MergeKey(word[i], word[i + 1])];
    }
  }

  return frequencies;
}

void MergePairInWord(Word& word, const std::string& left, const std::string& right, const std::string& merged) {
  if (word.size() < 2) {
    return;
  }

  std::vector<std::string> updated;
  updated.reserve(word.size());

  for (std::size_t i = 0; i < word.size(); ++i) {
    if (i + 1 < word.size() && word[i] == left && word[i + 1] == right) {
      updated.push_back(merged);
      ++i;
    } else {
      updated.push_back(word[i]);
    }
  }

  word = std::move(updated);
}

void CollectSymbols(const std::vector<Word>& words, Vocabulary& vocabulary) {
  for (const Word& word : words) {
    for (const std::string& symbol : word) {
      vocabulary.AddToken(symbol);
    }
  }
}

} // namespace

BpeTokenizer::BpeTokenizer() {
  vocabulary_.AddToken(kWordStartToken);
}

Status BpeTokenizer::Train(const std::string& corpus, std::size_t targetVocabSize) {
  if (targetVocabSize < kSpecialTokenCount + 1) {
    return Status::Fail(ErrorCode::InvalidArgument, "target vocabulary size is too small");
  }

  vocabulary_ = Vocabulary();
  vocabulary_.AddToken(kWordStartToken);
  merges_.clear();
  mergeRank_.clear();

  std::vector<Word> words = SplitCorpusIntoWords(corpus);
  if (words.empty()) {
    return Status::Fail(ErrorCode::InvalidArgument, "training corpus is empty");
  }

  CollectSymbols(words, vocabulary_);

  while (vocabulary_.Size() < targetVocabSize) {
    const auto frequencies = CountPairFrequencies(words);
    if (frequencies.empty()) {
      break;
    }

    const auto best = std::max_element(
        frequencies.begin(), frequencies.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });

    if (best->second < 2) {
      break;
    }

    const auto delimiter = best->first.find('\x1F');
    const std::string left = best->first.substr(0, delimiter);
    const std::string right = best->first.substr(delimiter + 1);
    const std::string merged = left + right;

    for (Word& word : words) {
      MergePairInWord(word, left, right, merged);
    }

    vocabulary_.AddToken(merged);
    merges_.push_back({left, right});
    mergeRank_[MergeKey(left, right)] = static_cast<int>(merges_.size()) - 1;
  }

  return Status::Ok();
}

int BpeTokenizer::FindBestMerge(const std::vector<std::string>& symbols) const {
  int bestRank = -1;
  int bestIndex = -1;

  for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
    const std::string key = MergeKey(symbols[i], symbols[i + 1]);
    const auto it = mergeRank_.find(key);
    if (it == mergeRank_.end()) {
      continue;
    }

    if (bestRank == -1 || it->second < bestRank) {
      bestRank = it->second;
      bestIndex = static_cast<int>(i);
    }
  }

  return bestIndex;
}

std::vector<std::string> BpeTokenizer::ApplyBpe(const std::vector<std::string>& symbols) const {
  if (symbols.size() < 2) {
    return symbols;
  }

  std::vector<std::string> current = symbols;

  while (current.size() >= 2) {
    const int mergeIndex = FindBestMerge(current);
    if (mergeIndex < 0) {
      break;
    }

    const MergePair& rule = merges_[static_cast<std::size_t>(
        mergeRank_.at(MergeKey(current[static_cast<std::size_t>(mergeIndex)], current[static_cast<std::size_t>(mergeIndex) + 1])))];
    const std::string merged = rule.first + rule.second;

    std::vector<std::string> next;
    next.reserve(current.size());

    for (std::size_t i = 0; i < current.size();) {
      if (static_cast<int>(i) == mergeIndex) {
        next.push_back(merged);
        i += 2;
      } else {
        next.push_back(current[i]);
        ++i;
      }
    }

    current = std::move(next);
  }

  return current;
}

std::vector<TokenId> BpeTokenizer::Encode(const std::string& text) const {
  std::vector<TokenId> ids;
  std::istringstream stream(text);
  std::string word;

  while (stream >> word) {
    std::vector<std::string> symbols;
    symbols.push_back(kWordStartToken);
    for (const char ch : word) {
      symbols.emplace_back(1, ch);
    }

    const std::vector<std::string> tokens = ApplyBpe(symbols);
    ids.reserve(ids.size() + tokens.size());
    for (const std::string& token : tokens) {
      ids.push_back(vocabulary_.GetId(token));
    }
  }

  return ids;
}

std::vector<TokenId> BpeTokenizer::EncodeWithSpecialTokens(const std::string& text, bool addBos, bool addEos) const {
  std::vector<TokenId> ids;

  if (addBos) {
    ids.push_back(kBosTokenId);
  }

  const std::vector<TokenId> encoded = Encode(text);
  ids.insert(ids.end(), encoded.begin(), encoded.end());

  if (addEos) {
    ids.push_back(kEosTokenId);
  }

  return ids;
}

std::string BpeTokenizer::Decode(const std::vector<TokenId>& ids) const {
  std::string text;

  for (const TokenId id : ids) {
    if (id == kPadTokenId || id == kBosTokenId || id == kEosTokenId || id == kUnkTokenId) {
      continue;
    }

    if (id >= vocabulary_.Size()) {
      continue;
    }

    const std::string& token = vocabulary_.GetToken(id);
    if (token == kWordStartToken) {
      if (!text.empty()) {
        text.push_back(' ');
      }
      continue;
    }

    text += token;
  }

  return text;
}

Status BpeTokenizer::Save(const std::string& directory) const {
  namespace fs = std::filesystem;

  std::error_code error;
  fs::create_directories(directory, error);
  if (error) {
    return Status::Fail(ErrorCode::IoError, "cannot create tokenizer directory: " + directory);
  }

  const Status vocabStatus = vocabulary_.Save((fs::path(directory) / "vocab.txt").string());
  if (!vocabStatus.IsOk()) {
    return vocabStatus;
  }

  std::ofstream merges((fs::path(directory) / "merges.txt").string());
  if (!merges) {
    return Status::Fail(ErrorCode::IoError, "cannot write merges file");
  }

  for (const MergePair& merge : merges_) {
    merges << merge.first << '\t' << merge.second << '\n';
  }

  return Status::Ok();
}

Result<BpeTokenizer> BpeTokenizer::Load(const std::string& directory) {
  namespace fs = std::filesystem;

  const fs::path vocabPath = fs::path(directory) / "vocab.txt";
  const fs::path mergesPath = fs::path(directory) / "merges.txt";

  auto vocabulary = Vocabulary::Load(vocabPath.string());
  if (!vocabulary.IsOk()) {
    return Result<BpeTokenizer>::Fail(vocabulary.GetError().code, vocabulary.GetError().message);
  }

  std::ifstream merges(mergesPath.string());
  if (!merges) {
    return Result<BpeTokenizer>::Fail(ErrorCode::IoError, "cannot read merges file");
  }

  BpeTokenizer tokenizer;
  tokenizer.vocabulary_ = std::move(vocabulary.Value());

  std::string line;
  while (std::getline(merges, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }

    const auto delimiter = line.find('\t');
    if (delimiter == std::string::npos) {
      return Result<BpeTokenizer>::Fail(ErrorCode::InvalidArgument, "invalid merge line");
    }

    const std::string left = line.substr(0, delimiter);
    const std::string right = line.substr(delimiter + 1);
    tokenizer.merges_.push_back({left, right});
    tokenizer.mergeRank_[MergeKey(left, right)] = static_cast<int>(tokenizer.merges_.size()) - 1;
  }

  return Result<BpeTokenizer>::Ok(std::move(tokenizer));
}

} // namespace llm
