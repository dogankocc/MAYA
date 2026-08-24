#pragma once

#include <optional>
#include <random>
#include <string>

#include "llm/core/status.hpp"
#include "llm/inference/inference_engine.hpp"
#include "llm/inference/sampling/generator.hpp"
#include "llm/inference/sampling/sampler.hpp"
#include "llm/model/transformer.hpp"
#include "llm/tokenizer/bpe_tokenizer.hpp"

namespace llm::cli {

class ChatSession {
public:
  ChatSession();

  [[nodiscard]] Status Load(const std::string& modelPath, const std::string& tokenizerPath);

  [[nodiscard]] Result<std::string> Complete(const std::string& prompt, std::size_t maxNewTokens, std::mt19937& rng);

  void ResetConversation();

  [[nodiscard]] bool IsLoaded() const { return loaded_; }

  [[nodiscard]] inference::SamplingConfig& Sampling() { return sampling_; }

  [[nodiscard]] const inference::SamplingConfig& Sampling() const { return sampling_; }

private:
  bool loaded_ = false;
  std::optional<model::TransformerModel> model_;
  BpeTokenizer tokenizer_;
  inference::SamplingConfig sampling_;
  std::optional<inference::InferenceEngine> engine_;
  std::optional<inference::TextGenerator> generator_;
};

} // namespace llm::cli
