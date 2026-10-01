#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
#include <unordered_map>

#include "llm/tokenizer/bpe_tokenizer.hpp"

TEST(BpeTokenizerTest, TrainBuildsMerges) {
  llm::BpeTokenizer tokenizer;
  const std::string corpus = "araba araba araba araba araba araba";
  ASSERT_TRUE(tokenizer.Train(corpus, 32).IsOk());
  EXPECT_GT(tokenizer.GetVocabulary().Size(), llm::kSpecialTokenCount);
}

TEST(BpeTokenizerTest, EncodeDecodeRoundTrip) {
  llm::BpeTokenizer tokenizer;
  const std::string corpus =
      "araba araba araba test test test model model model model";
  ASSERT_TRUE(tokenizer.Train(corpus, 64).IsOk());

  const std::string text = "araba";
  const std::vector<llm::TokenId> ids = tokenizer.Encode(text);
  EXPECT_FALSE(ids.empty());

  const std::string decoded = tokenizer.Decode(ids);
  EXPECT_NE(decoded.find("araba"), std::string::npos);
}

TEST(BpeTokenizerTest, TurkishUtf8TextRoundTripsExactly) {
  llm::BpeTokenizer tokenizer;
  const std::string text = "Ben senin yaratt" "\xC4\xB1" "n bir yapay zeka asistan" "\xC4\xB1" "y" "\xC4\xB1" "m.";
  const std::string TurkishWord = "nas" "\xC4\xB1" "ls" "\xC4\xB1" "n";
  ASSERT_TRUE(tokenizer.Train(text + " " + TurkishWord + " merhaba", 128).IsOk());

  EXPECT_EQ(tokenizer.Decode(tokenizer.Encode(text)), text);
  EXPECT_EQ(tokenizer.Decode(tokenizer.Encode(TurkishWord)), TurkishWord);
}

TEST(BpeTokenizerTest, DecodeKeepsValidByteSequencesAndRepairsInvalidUtf8) {
  llm::BpeTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.Train("\xC3\xA9", 32).IsOk());

  const std::vector<llm::TokenId> encoded = tokenizer.Encode("\xC3\xA9");
  ASSERT_EQ(encoded.size(), 3U);  // word-start marker plus one reversible symbol for each UTF-8 byte

  EXPECT_EQ(tokenizer.Decode(encoded), "\xC3\xA9");
  EXPECT_EQ(tokenizer.Decode({encoded.back()}), "\xEF\xBF\xBD");
  EXPECT_EQ(tokenizer.Decode({llm::kUnkTokenId}), "\xEF\xBF\xBD");
}

TEST(BpeTokenizerTest, ByteFallbackRoundTripsCharactersAbsentFromTrainingCorpus) {
  llm::BpeTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.Train("ascii corpus", 300).IsOk());

  const std::string unseenCharacter = "\xF0\x9F\xA6\x8A";
  const std::vector<llm::TokenId> ids = tokenizer.Encode(unseenCharacter);
  ASSERT_FALSE(ids.empty());
  EXPECT_TRUE(std::all_of(ids.begin(), ids.end(), [](const llm::TokenId id) { return id != llm::kUnkTokenId; }));
  EXPECT_EQ(tokenizer.Decode(ids), unseenCharacter);
}

TEST(BpeTokenizerTest, MergesMostFrequentPairFirstAndReportsProgress) {
  llm::BpeTokenizer tokenizer;
  const std::string corpus = "ab ab ab ab ab ab cd cd cd xyz";
  std::vector<std::size_t> reported;
  const auto progress = [&reported](const std::size_t vocab, const std::size_t) { reported.push_back(vocab); };
  constexpr std::size_t base = llm::kSpecialTokenCount + 1 + 256;
  ASSERT_TRUE(tokenizer.Train(corpus, base + 1, progress).IsOk());

  const auto& vocab = tokenizer.GetVocabulary();
  EXPECT_EQ(vocab.Size(), base + 1);
  EXPECT_NE(vocab.GetId(std::string(llm::kWordStartToken) + "a"), llm::kUnkTokenId);

  const auto ids = tokenizer.Encode("ab ab");
  EXPECT_EQ(ids.size(), 4U);
  EXPECT_EQ(tokenizer.Decode(ids), "ab ab");
  EXPECT_TRUE(reported.empty());
}

namespace {

// Naive reference BPE with the same tie-break (higher count, then smaller (left,right) ids).
std::vector<std::pair<std::string, std::string>> ReferenceMerges(const std::string& corpus, const int mergeCount) {
  std::vector<std::string> symbols{llm::kWordStartToken};
  std::unordered_map<std::string, int> ids{{llm::kWordStartToken, 0}};
  const auto symbolId = [&](const std::string& s) {
    const auto [it, inserted] = ids.emplace(s, static_cast<int>(symbols.size()));
    if (inserted) {
      symbols.push_back(s);
    }
    return it->second;
  };

  // The trainer assigns byte-symbol IDs in ascending byte order before reading words.
  for (int byte = 0; byte < 256; ++byte) {
    const bool direct = (byte >= 0x21 && byte <= 0x7E) || (byte >= 0xA1 && byte <= 0xAC) ||
                        (byte >= 0xAE && byte <= 0xFF);
    int codePoint = byte;
    if (!direct) {
      int extra = 0;
      for (int prior = 0; prior < byte; ++prior) {
        const bool priorDirect = (prior >= 0x21 && prior <= 0x7E) || (prior >= 0xA1 && prior <= 0xAC) ||
                                 (prior >= 0xAE && prior <= 0xFF);
        if (!priorDirect) {
          ++extra;
        }
      }
      codePoint = 256 + extra;
    }
    std::string symbol;
    if (codePoint <= 0x7F) {
      symbol.push_back(static_cast<char>(codePoint));
    } else if (codePoint <= 0x7FF) {
      symbol.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
      symbol.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else {
      symbol.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
      symbol.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
      symbol.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
    (void)symbolId(symbol);
  }

  std::vector<std::vector<int>> words;
  std::istringstream stream(corpus);
  std::string token;
  while (stream >> token) {
    std::vector<int> word{0};
    for (const char ch : token) {
      word.push_back(symbolId(std::string(1, ch)));
    }
    words.push_back(word);
  }

  std::vector<std::pair<std::string, std::string>> merges;
  for (int m = 0; m < mergeCount; ++m) {
    std::map<std::pair<int, int>, long long> counts;
    for (const auto& word : words) {
      for (std::size_t i = 0; i + 1 < word.size(); ++i) {
        ++counts[{word[i], word[i + 1]}];
      }
    }
    std::pair<int, int> best{-1, -1};
    long long bestCount = 1;
    for (const auto& [pair, count] : counts) {
      if (count > bestCount) {
        bestCount = count;
        best = pair;
      }
    }
    if (best.first < 0) {
      break;
    }
    const int merged = symbolId(symbols[best.first] + symbols[best.second]);
    for (auto& word : words) {
      std::vector<int> next;
      for (std::size_t i = 0; i < word.size(); ++i) {
        if (i + 1 < word.size() && word[i] == best.first && word[i + 1] == best.second) {
          next.push_back(merged);
          ++i;
        } else {
          next.push_back(word[i]);
        }
      }
      word = std::move(next);
    }
    merges.emplace_back(symbols[best.first], symbols[best.second]);
  }
  return merges;
}

std::vector<std::pair<std::string, std::string>> SavedMerges(const llm::BpeTokenizer& tokenizer) {
  const auto dir = std::filesystem::temp_directory_path() / "maya_bpe_ref_test";
  std::filesystem::create_directories(dir);
  EXPECT_TRUE(tokenizer.Save(dir.string()).IsOk());
  std::ifstream in((dir / "merges.txt").string());
  std::vector<std::pair<std::string, std::string>> merges;
  std::string line;
  while (std::getline(in, line)) {
    const auto tab = line.find('\t');
    merges.emplace_back(line.substr(0, tab), line.substr(tab + 1));
  }
  in.close();
  std::filesystem::remove_all(dir);
  return merges;
}

} // namespace

TEST(BpeTokenizerTest, IncrementalTrainerMatchesNaiveReference) {
  std::string corpus;
  std::mt19937 rng(7);
  const char* pool[] = {"kod", "kodlama", "model", "modelleme", "veri", "verisi", "aaa", "aaaa", "abab", "ababab",
                        "xyz", "xyzxyz", "merhaba", "dunya", "test", "tester", "testing"};
  for (int i = 0; i < 400; ++i) {
    corpus += pool[rng() % std::size(pool)];
    corpus += ' ';
  }

  constexpr int kMerges = 40;
  const auto reference = ReferenceMerges(corpus, kMerges);
  ASSERT_EQ(reference.size(), static_cast<std::size_t>(kMerges));

  llm::BpeTokenizer tokenizer;
  constexpr std::size_t baseVocab = llm::kSpecialTokenCount + 1 + 256;
  ASSERT_TRUE(tokenizer.Train(corpus, baseVocab + kMerges).IsOk());

  const auto saved = SavedMerges(tokenizer);
  ASSERT_GE(saved.size(), reference.size());
  EXPECT_TRUE(std::equal(reference.begin(), reference.end(), saved.begin()));
}

TEST(BpeTokenizerTest, LargeRepetitiveCorpusTrainsQuickly) {
  std::string corpus;
  for (int repeat = 0; repeat < 200; ++repeat) {
    corpus += "merhaba dunya bu bir deneme metni kod yazmak guzel ";
    for (int i = 0; i < 300; ++i) {
      corpus += "kelime" + std::to_string(i * 7919 % 1000) + ' ';
    }
  }
  llm::BpeTokenizer tokenizer;
  std::size_t reports = 0;
  ASSERT_TRUE(tokenizer.Train(corpus, 512, [&reports](std::size_t, std::size_t) { ++reports; }).IsOk());
  EXPECT_GT(reports, 0U);
  EXPECT_EQ(tokenizer.Decode(tokenizer.Encode("kod yazmak guzel")), "kod yazmak guzel");
  EXPECT_LE(tokenizer.Encode("kelime").size(), 2U);
  EXPECT_LT(tokenizer.Encode("merhaba").size(), 8U);
}

TEST(BpeTokenizerTest, EncodeWithSpecialTokens) {
  llm::BpeTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.Train("hello hello hello", 32).IsOk());

  const auto ids = tokenizer.EncodeWithSpecialTokens("hello", true, true);
  ASSERT_GE(ids.size(), 3U);
  EXPECT_EQ(ids.front(), llm::kBosTokenId);
  EXPECT_EQ(ids.back(), llm::kEosTokenId);
}

TEST(BpeTokenizerTest, SaveAndLoadRoundTrip) {
  llm::BpeTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.Train("merhaba dunya merhaba dunya", 48).IsOk());

  const std::string directory = "test_tokenizer";
  ASSERT_TRUE(tokenizer.Save(directory).IsOk());

  const auto loaded = llm::BpeTokenizer::Load(directory);
  ASSERT_TRUE(loaded.IsOk());

  const std::string text = "merhaba";
  const auto& loadedTokenizer = loaded.Value();
  const std::string decoded = loadedTokenizer.Decode(loadedTokenizer.Encode(text));
  EXPECT_NE(decoded.find("merhaba"), std::string::npos);

  std::remove((directory + "/vocab.txt").c_str());
  std::remove((directory + "/merges.txt").c_str());
#ifdef _WIN32
  _rmdir(directory.c_str());
#else
  rmdir(directory.c_str());
#endif
}
