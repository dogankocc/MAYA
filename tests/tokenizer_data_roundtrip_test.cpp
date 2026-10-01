#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

#include "llm/tokenizer/bpe_tokenizer.hpp"

TEST(TokenizerDataTest, FreshByteLevelTokenizerRoundTripsTurkishAndUnseenUnicode) {
  const std::string trainingText = "Merhaba nas\xC4\xB1ls\xC4\xB1n? Ben senin yaratt\xC4\xB1\xC4\x9F\xC4\xB1n bir yapay zeka asistan\xC4\xB1y\xC4\xB1m.";
  llm::BpeTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.Train(trainingText, 512).IsOk());

  const std::string examples[] = {
      "Merhaba nas\xC4\xB1ls\xC4\xB1n?",
      "Ben senin yaratt\xC4\xB1\xC4\x9F\xC4\xB1n bir yapay zeka asistan\xC4\xB1y\xC4\xB1m.",
      "Eğitimde olmayan emoji: \xF0\x9F\xA6\x8A",
  };
  for (const std::string& text : examples) {
    const std::vector<llm::TokenId> ids = tokenizer.Encode(text);
    ASSERT_TRUE(std::all_of(ids.begin(), ids.end(), [](const llm::TokenId id) { return id != llm::kUnkTokenId; }));
    EXPECT_EQ(tokenizer.Decode(ids), text);
  }

  const auto withBosEos = tokenizer.EncodeWithSpecialTokens(examples[0], true, true);
  ASSERT_GE(withBosEos.size(), 3U);
  EXPECT_EQ(withBosEos.front(), llm::kBosTokenId);
  EXPECT_EQ(withBosEos.back(), llm::kEosTokenId);
}

TEST(TokenizerDataTest, RejectsVocabularyWithoutCompleteByteLevelAlphabet) {
  namespace fs = std::filesystem;
  const fs::path directory = fs::temp_directory_path() / "maya_incomplete_tokenizer_test";
  fs::remove_all(directory);
  fs::create_directories(directory);

  llm::Vocabulary vocabulary;
  ASSERT_TRUE(vocabulary.Save((directory / "vocab.txt").string()).IsOk());
  {
    std::ofstream merges(directory / "merges.txt");
    ASSERT_TRUE(merges.good());
  }

  const auto loaded = llm::BpeTokenizer::Load(directory.string());
  EXPECT_FALSE(loaded.IsOk());
  fs::remove_all(directory);
}
