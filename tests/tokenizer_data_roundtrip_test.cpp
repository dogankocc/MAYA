#include <gtest/gtest.h>
#include <iostream>
#include <filesystem>
#include <stdlib.h>

#include "llm/tokenizer/bpe_tokenizer.hpp"

namespace {

namespace fs = std::filesystem;

// Test çalışma dizininden proje köküne ulaş
fs::path FindProjectRoot() {
  fs::path exeDir = fs::current_path();
  
  // Önce mevcut dizinde kontrol et
  if (fs::exists(exeDir / "tokenizer_data")) {
    return exeDir;
  }
  
  // Üst dizinlere çık - tipik: out/build/x64-release/tests -> projedir
  for (int up = 0; up < 5; ++up) {
    fs::path candidate = exeDir;
    for (int i = 0; i < up; ++i) {
      candidate = candidate.parent_path();
    }
    if (fs::exists(candidate / "tokenizer_data")) {
      return candidate;
    }
    if (fs::exists(candidate / "MAYA")) {
      // MAYA.exe'nin olduğu klasörden bir üst
      if (fs::exists(candidate.parent_path() / "tokenizer_data")) {
        return candidate.parent_path();
      }
    }
  }
  
  // Environment variable veya CMAKE'den gelen yol
  const char* projectDir = getenv("PROJECT_SOURCE_DIR");
  if (projectDir && fs::exists(fs::path(projectDir) / "tokenizer_data")) {
    return fs::path(projectDir);
  }
  
  return exeDir;
}

} // namespace

TEST(TokenizerDataTest, EncodeDecodeRoundTripTurkish) {
  const fs::path projectRoot = FindProjectRoot();
  const fs::path tokenizerDir = projectRoot / "tokenizer_data";
  
  std::cout << "\n=== Tokenizer Test ===" << std::endl;
  std::cout << "Current dir: " << fs::current_path() << std::endl;
  std::cout << "Project root: " << projectRoot << std::endl;
  std::cout << "Tokenizer dir: " << tokenizerDir << std::endl;
  std::cout << "Exists: " << std::boolalpha << fs::exists(tokenizerDir) << std::endl;
  
  ASSERT_TRUE(fs::exists(tokenizerDir))
      << "Tokenizer directory not found. Tried: " << tokenizerDir;

  const auto loaded = llm::BpeTokenizer::Load(tokenizerDir.string());
  ASSERT_TRUE(loaded.IsOk()) << "Failed to load tokenizer: " << loaded.GetError().message;

  const llm::BpeTokenizer& tokenizer = loaded.Value();
  std::cout << "Vocab size: " << tokenizer.GetVocabulary().Size() << std::endl;

  // Test 1: Basit turkce metin
  {
    const std::string text = "Merhaba nasılsın?";
    std::cout << "\n--- Test: " << text << " ---" << std::endl;

    const std::vector<llm::TokenId> ids = tokenizer.Encode(text);
    std::cout << "Encoded (" << ids.size() << "): ";
    for (const llm::TokenId id : ids) {
      std::cout << id << " ";
    }
    std::cout << std::endl;
    
    // Her token'ın decode edilmiş halini göster
    std::cout << "Tokens: ";
    for (const llm::TokenId id : ids) {
      if (id < tokenizer.GetVocabulary().Size()) {
        const std::string token = tokenizer.GetVocabulary().GetToken(id);
        std::cout << "[" << token << "] ";
      } else {
        std::cout << "[UNK:" << id << "] ";
      }
    }
    std::cout << std::endl;

    const std::string decoded = tokenizer.Decode(ids);
    std::cout << "Decoded:  [" << decoded << "]" << std::endl;
    std::cout << "Original: [" << text << "]" << std::endl;
    
    EXPECT_EQ(decoded, text) << "Round-trip failed!";
  }

  // Test 2: Eğitim verisinden tipik bir cevap
  {
    const std::string text = "Ben senin yarattığın bir yapay zeka asistanıyım.";
    std::cout << "\n--- Test: " << text << " ---" << std::endl;

    const std::vector<llm::TokenId> ids = tokenizer.Encode(text);
    std::cout << "Encoded (" << ids.size() << "): ";
    for (const llm::TokenId id : ids) {
      std::cout << id << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Tokens: ";
    for (const llm::TokenId id : ids) {
      if (id < tokenizer.GetVocabulary().Size()) {
        const std::string token = tokenizer.GetVocabulary().GetToken(id);
        std::cout << "[" << token << "] ";
      } else {
        std::cout << "[UNK:" << id << "] ";
      }
    }
    std::cout << std::endl;

    const std::string decoded = tokenizer.Decode(ids);
    std::cout << "Decoded:  [" << decoded << "]" << std::endl;
    std::cout << "Original: [" << text << "]" << std::endl;
    
    EXPECT_EQ(decoded, text) << "Round-trip failed!";
  }
  
  // Test 3: EncodeWithSpecialTokens ile inference formatı
  {
    const std::string text = "Merhaba";
    std::cout << "\n--- Test: EncodeWithSpecialTokens('" << text << "') ---" << std::endl;
    
    const std::vector<llm::TokenId> withBosEos = tokenizer.EncodeWithSpecialTokens(text, true, true);
    std::cout << "With BOS/EOS (" << withBosEos.size() << "): ";
    for (const llm::TokenId id : withBosEos) {
      std::cout << id << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Tokens: ";
    for (const llm::TokenId id : withBosEos) {
      if (id == llm::kBosTokenId) {
        std::cout << "[BOS] ";
      } else if (id == llm::kEosTokenId) {
        std::cout << "[EOS] ";
      } else if (id < tokenizer.GetVocabulary().Size()) {
        const std::string token = tokenizer.GetVocabulary().GetToken(id);
        std::cout << "[" << token << "] ";
      } else {
        std::cout << "[UNK:" << id << "] ";
      }
    }
    std::cout << std::endl;
  }
}
