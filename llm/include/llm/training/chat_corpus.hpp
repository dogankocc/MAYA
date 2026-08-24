#pragma once

#include <string>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/core/types.hpp"
#include "llm/tokenizer/bpe_tokenizer.hpp"

namespace llm::training {

struct DialogueSample {
  std::string intent;
  std::string system;
  std::string prompt;
  std::string response;
};

[[nodiscard]] Result<std::vector<DialogueSample>> LoadDialogueCorpus(const std::string& path);

[[nodiscard]] Result<std::vector<DialogueSample>> LoadDialogueCorpusDirectory(const std::string& directoryPath);

[[nodiscard]] std::vector<DialogueSample> DefaultDialogueCorpus();

[[nodiscard]] std::vector<DialogueSample> MergeDialogueCorpora(std::vector<DialogueSample> base,
                                                               const std::vector<DialogueSample>& extra);

[[nodiscard]] std::string BuildTokenizerCorpus(const std::vector<DialogueSample>& samples);

[[nodiscard]] std::vector<std::vector<TokenId>> BuildTrainingBatches(const BpeTokenizer& tokenizer,
                                                                     const std::vector<DialogueSample>& samples,
                                                                     std::size_t maxSeqLen);

} // namespace llm::training
