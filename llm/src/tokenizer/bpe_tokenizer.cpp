#include "llm/tokenizer/bpe_tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <queue>
#include <string_view>
#include <unordered_map>

namespace llm {

namespace {

constexpr std::size_t kProgressEveryMerges = 64;
constexpr std::size_t kMaxEncodeCacheEntries = 500000;
constexpr std::size_t kWordStartTokenLength = 3;

std::string MergeKey(const std::string& left, const std::string& right) {
  return left + "\x1F" + right;
}

[[nodiscard]] bool IsWordSeparator(const char ch) {
  return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\v' || ch == '\f';
}

template <typename Fn>
void ForEachWord(const std::string& text, Fn&& fn) {
  std::size_t start = 0;
  while (start < text.size()) {
    while (start < text.size() && IsWordSeparator(text[start])) {
      ++start;
    }
    std::size_t end = start;
    while (end < text.size() && !IsWordSeparator(text[end])) {
      ++end;
    }
    if (end > start) {
      fn(std::string_view(text).substr(start, end - start));
    }
    start = end;
  }
}

// Incremental BPE trainer: words are de-duplicated with counts, pair frequencies are
// maintained incrementally and the best pair is picked from a lazily-validated heap.
class BpeTrainer {
public:
  using ProgressFn = BpeTokenizer::TrainProgressFn;
  using SymbolId = std::uint32_t;
  using PairKey = std::uint64_t;

  BpeTrainer(const std::string& corpus, Vocabulary& vocabulary) : vocabulary_(vocabulary) {
    BuildWords(corpus);
  }

  [[nodiscard]] bool Empty() const { return words_.empty(); }

  void Run(const std::size_t targetVocabSize, const ProgressFn& progress,
           const std::function<void(const std::string&, const std::string&)>& onMerge) {
    for (const std::string& symbol : symbols_) {
      vocabulary_.AddToken(symbol);
    }
    InitializePairStats();

    std::size_t mergesSinceReport = 0;
    while (vocabulary_.Size() < targetVocabSize) {
      const auto best = PopBestPair();
      if (!best.has_value() || best->count < 2) {
        break;
      }
      const auto [left, right] = SplitKey(best->key);
      const SymbolId mergedId = AddSymbol(symbols_[left] + symbols_[right]);
      MergeAllOccurrences(best->key, left, right, mergedId);

      vocabulary_.AddToken(symbols_[mergedId]);
      onMerge(symbols_[left], symbols_[right]);

      if (progress && ++mergesSinceReport >= kProgressEveryMerges) {
        mergesSinceReport = 0;
        progress(vocabulary_.Size(), targetVocabSize);
      }
    }
  }

private:
  struct TrainWord {
    std::vector<SymbolId> symbols;
    std::uint64_t count = 0;
    std::uint32_t lastMerge = 0;
  };

  struct HeapEntry {
    std::uint64_t count;
    PairKey key;
    bool operator<(const HeapEntry& other) const {
      return count != other.count ? count < other.count : key > other.key;
    }
  };

  static PairKey MakeKey(const SymbolId left, const SymbolId right) {
    return (static_cast<PairKey>(left) << 32U) | right;
  }

  static std::pair<SymbolId, SymbolId> SplitKey(const PairKey key) {
    return {static_cast<SymbolId>(key >> 32U), static_cast<SymbolId>(key & 0xFFFFFFFFU)};
  }

  SymbolId AddSymbol(const std::string& symbol) {
    const auto [it, inserted] = symbolIds_.emplace(symbol, static_cast<SymbolId>(symbols_.size()));
    if (inserted) {
      symbols_.push_back(symbol);
    }
    return it->second;
  }

  void BuildWords(const std::string& corpus) {
    std::unordered_map<std::string_view, std::uint32_t> wordIndex;
    const SymbolId startId = AddSymbol(kWordStartToken);
    ForEachWord(corpus, [&](const std::string_view token) {
      const auto [it, inserted] = wordIndex.emplace(token, static_cast<std::uint32_t>(words_.size()));
      if (!inserted) {
        ++words_[it->second].count;
        return;
      }
      TrainWord word;
      word.count = 1;
      word.symbols.reserve(token.size() + 1);
      word.symbols.push_back(startId);
      for (const char ch : token) {
        word.symbols.push_back(AddSymbol(std::string(1, ch)));
      }
      words_.push_back(std::move(word));
    });
  }

  void InitializePairStats() {
    for (std::uint32_t wordIdx = 0; wordIdx < words_.size(); ++wordIdx) {
      const TrainWord& word = words_[wordIdx];
      for (std::size_t i = 0; i + 1 < word.symbols.size(); ++i) {
        const PairKey key = MakeKey(word.symbols[i], word.symbols[i + 1]);
        pairCounts_[key] += word.count;
        pairWords_[key].push_back(wordIdx);
      }
    }
    for (const auto& [key, count] : pairCounts_) {
      heap_.push(HeapEntry{count, key});
    }
  }

  std::optional<HeapEntry> PopBestPair() {
    while (!heap_.empty()) {
      const HeapEntry entry = heap_.top();
      heap_.pop();
      const auto it = pairCounts_.find(entry.key);
      if (it != pairCounts_.end() && it->second == entry.count && entry.count > 0) {
        return entry;
      }
    }
    return std::nullopt;
  }

  void AdjustPair(const PairKey key, const std::int64_t delta, const std::uint32_t wordIdx, const bool indexWord) {
    auto& count = pairCounts_[key];
    count = static_cast<std::uint64_t>(static_cast<std::int64_t>(count) + delta);
    if (indexWord) {
      pairWords_[key].push_back(wordIdx);
    }
    touched_.push_back(key);
  }

  void MergeAllOccurrences(const PairKey key, const SymbolId left, const SymbolId right, const SymbolId mergedId) {
    ++mergeStamp_;
    touched_.clear();
    std::vector<std::uint32_t> candidates = std::move(pairWords_[key]);
    pairWords_.erase(key);
    pairCounts_.erase(key);

    for (const std::uint32_t wordIdx : candidates) {
      TrainWord& word = words_[wordIdx];
      if (word.lastMerge == mergeStamp_) {
        continue;
      }
      word.lastMerge = mergeStamp_;
      RewriteWord(word, wordIdx, key, left, right, mergedId);
    }

    std::sort(touched_.begin(), touched_.end());
    touched_.erase(std::unique(touched_.begin(), touched_.end()), touched_.end());
    for (const PairKey touchedKey : touched_) {
      const auto it = pairCounts_.find(touchedKey);
      if (it == pairCounts_.end()) {
        continue;
      }
      if (it->second == 0) {
        pairCounts_.erase(it);
        pairWords_.erase(touchedKey);
      } else {
        heap_.push(HeapEntry{it->second, touchedKey});
      }
    }
  }

  void RewriteWord(TrainWord& word, const std::uint32_t wordIdx, const PairKey key, const SymbolId left,
                   const SymbolId right, const SymbolId mergedId) {
    const std::vector<SymbolId>& old = word.symbols;
    bool contains = false;
    for (std::size_t i = 0; i + 1 < old.size(); ++i) {
      if (old[i] == left && old[i + 1] == right) {
        contains = true;
        break;
      }
    }
    if (!contains) {
      return;
    }

    // Only pairs adjacent to a merge site change; everything else keeps its count.
    const auto delta = static_cast<std::int64_t>(word.count);
    const auto remove = [&](const SymbolId a, const SymbolId b) {
      const PairKey pair = MakeKey(a, b);
      if (pair != key) {
        AdjustPair(pair, -delta, wordIdx, false);
      }
    };
    const auto add = [&](const SymbolId a, const SymbolId b) { AdjustPair(MakeKey(a, b), delta, wordIdx, true); };

    std::vector<SymbolId> updated;
    updated.reserve(old.size());
    bool lastWasMerge = false;
    for (std::size_t i = 0; i < old.size();) {
      if (i + 1 < old.size() && old[i] == left && old[i + 1] == right) {
        if (i > 0 && !lastWasMerge) {
          remove(old[i - 1], left);
        }
        if (i + 2 < old.size()) {
          remove(right, old[i + 2]);
        }
        if (!updated.empty()) {
          add(updated.back(), mergedId);
        }
        updated.push_back(mergedId);
        lastWasMerge = true;
        i += 2;
      } else {
        if (lastWasMerge) {
          add(mergedId, old[i]);
        }
        updated.push_back(old[i]);
        lastWasMerge = false;
        ++i;
      }
    }
    word.symbols = std::move(updated);
  }

  Vocabulary& vocabulary_;
  std::vector<std::string> symbols_;
  std::unordered_map<std::string, SymbolId> symbolIds_;
  std::vector<TrainWord> words_;
  std::unordered_map<PairKey, std::uint64_t> pairCounts_;
  std::unordered_map<PairKey, std::vector<std::uint32_t>> pairWords_;
  std::priority_queue<HeapEntry> heap_;
  std::vector<PairKey> touched_;
  std::uint32_t mergeStamp_ = 0;
};

} // namespace

struct BpeTokenizer::EncodeCache {
  std::mutex mutex;
  std::unordered_map<std::string, std::vector<TokenId>> words;
};

BpeTokenizer::BpeTokenizer() : cache_(std::make_shared<EncodeCache>()) {
  vocabulary_.AddToken(kWordStartToken);
}

void BpeTokenizer::ResetCache() {
  cache_ = std::make_shared<EncodeCache>();
}

void BpeTokenizer::AddMerge(const std::string& left, const std::string& right) {
  merges_.push_back({left, right});
  mergeRank_[MergeKey(left, right)] = static_cast<int>(merges_.size()) - 1;
}

Status BpeTokenizer::Train(const std::string& corpus, std::size_t targetVocabSize, const TrainProgressFn& progress) {
  if (targetVocabSize < kSpecialTokenCount + 1) {
    return Status::Fail(ErrorCode::InvalidArgument, "target vocabulary size is too small");
  }

  vocabulary_ = Vocabulary();
  merges_.clear();
  mergeRank_.clear();
  ResetCache();

  BpeTrainer trainer(corpus, vocabulary_);
  if (trainer.Empty()) {
    vocabulary_.AddToken(kWordStartToken);
    return Status::Fail(ErrorCode::InvalidArgument, "training corpus is empty");
  }

  trainer.Run(targetVocabSize, progress,
              [this](const std::string& left, const std::string& right) { AddMerge(left, right); });
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

std::vector<TokenId> BpeTokenizer::EncodeWord(const std::string& word) const {
  {
    std::lock_guard<std::mutex> lock(cache_->mutex);
    const auto it = cache_->words.find(word);
    if (it != cache_->words.end()) {
      return it->second;
    }
  }

  std::vector<std::string> symbols;
  symbols.reserve(word.size() + 1);
  symbols.push_back(kWordStartToken);
  for (const char ch : word) {
    symbols.emplace_back(1, ch);
  }

  std::vector<TokenId> ids;
  for (const std::string& token : ApplyBpe(symbols)) {
    ids.push_back(vocabulary_.GetId(token));
  }

  std::lock_guard<std::mutex> lock(cache_->mutex);
  if (cache_->words.size() < kMaxEncodeCacheEntries) {
    cache_->words.emplace(word, ids);
  }
  return ids;
}

std::vector<TokenId> BpeTokenizer::Encode(const std::string& text) const {
  std::vector<TokenId> ids;
  ForEachWord(text, [&](const std::string_view word) {
    const std::vector<TokenId> tokens = EncodeWord(std::string(word));
    ids.insert(ids.end(), tokens.begin(), tokens.end());
  });
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
    if (token.starts_with(kWordStartToken)) {
      if (!text.empty()) {
        text.push_back(' ');
      }
      text.append(token, kWordStartTokenLength, std::string::npos);
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

    tokenizer.AddMerge(line.substr(0, delimiter), line.substr(delimiter + 1));
  }

  return Result<BpeTokenizer>::Ok(std::move(tokenizer));
}

} // namespace llm
