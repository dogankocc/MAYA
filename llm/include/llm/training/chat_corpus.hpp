#pragma once

#include <functional>
#include <cstddef>
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

using BatchProgressFn = std::function<void(std::size_t done, std::size_t total)>;

struct TrainingExample {
  std::vector<TokenId> tokens;
  // Token index of the first assistant response token; prompt tokens provide context only.
  std::size_t firstTargetToken = 1;
};

[[nodiscard]] std::vector<TrainingExample> BuildTrainingBatches(const BpeTokenizer& tokenizer,
                                                                const std::vector<DialogueSample>& samples,
                                                                std::size_t maxSeqLen,
                                                                const BatchProgressFn& progress = nullptr);

} // namespace llm::training
