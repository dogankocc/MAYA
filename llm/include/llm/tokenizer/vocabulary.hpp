#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/core/types.hpp"

namespace llm {

constexpr const char* kPadToken = "<PAD>";
constexpr const char* kUnkToken = "<UNK>";
constexpr const char* kBosToken = "<BOS>";
constexpr const char* kEosToken = "<EOS>";
constexpr const char* kWordStartToken = "\xE2\x96\x81"; // UTF-8: ▁ (sentencepiece-style word boundary)

constexpr TokenId kPadTokenId = 0;
constexpr TokenId kUnkTokenId = 1;
constexpr TokenId kBosTokenId = 2;
constexpr TokenId kEosTokenId = 3;
constexpr TokenId kSpecialTokenCount = 4;

class Vocabulary {
public:
  Vocabulary();

  [[nodiscard]] TokenId AddToken(const std::string& token);

  [[nodiscard]] bool Contains(const std::string& token) const;

  [[nodiscard]] TokenId GetId(const std::string& token) const;

  [[nodiscard]] const std::string& GetToken(TokenId id) const;

  [[nodiscard]] std::size_t Size() const { return idToToken_.size(); }

  void Clear();

  [[nodiscard]] Status Save(const std::string& path) const;

  [[nodiscard]] static Result<Vocabulary> Load(const std::string& path);

private:
  std::vector<std::string> idToToken_;
  std::unordered_map<std::string, TokenId> tokenToId_;
};

} // namespace llm
