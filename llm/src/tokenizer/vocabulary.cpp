#include "llm/tokenizer/vocabulary.hpp"

#include <fstream>
#include <stdexcept>

namespace llm {

Vocabulary::Vocabulary() {
  AddToken(kPadToken);
  AddToken(kUnkToken);
  AddToken(kBosToken);
  AddToken(kEosToken);
}

TokenId Vocabulary::AddToken(const std::string& token) {
  const auto existing = tokenToId_.find(token);
  if (existing != tokenToId_.end()) {
    return existing->second;
  }

  const TokenId id = static_cast<TokenId>(idToToken_.size());
  idToToken_.push_back(token);
  tokenToId_[token] = id;
  return id;
}

bool Vocabulary::Contains(const std::string& token) const {
  return tokenToId_.contains(token);
}

TokenId Vocabulary::GetId(const std::string& token) const {
  const auto it = tokenToId_.find(token);
  if (it == tokenToId_.end()) {
    return kUnkTokenId;
  }
  return it->second;
}

const std::string& Vocabulary::GetToken(TokenId id) const {
  if (id >= idToToken_.size()) {
    throw std::out_of_range("invalid token id");
  }
  return idToToken_[id];
}

void Vocabulary::Clear() {
  idToToken_.clear();
  tokenToId_.clear();
}

Status Vocabulary::Save(const std::string& path) const {
  std::ofstream output(path);
  if (!output) {
    return Status::Fail(ErrorCode::IoError, "cannot write vocabulary: " + path);
  }

  for (const std::string& token : idToToken_) {
    output << token << '\n';
  }

  return Status::Ok();
}

Result<Vocabulary> Vocabulary::Load(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    return Result<Vocabulary>::Fail(ErrorCode::IoError, "cannot read vocabulary: " + path);
  }

  Vocabulary vocabulary;
  vocabulary.Clear();

  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    vocabulary.AddToken(line);
  }

  if (vocabulary.Size() < kSpecialTokenCount) {
    return Result<Vocabulary>::Fail(ErrorCode::InvalidArgument, "vocabulary missing special tokens");
  }

  return Result<Vocabulary>::Ok(std::move(vocabulary));
}

} // namespace llm
