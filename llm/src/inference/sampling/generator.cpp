#include "llm/inference/sampling/generator.hpp"

namespace llm::inference {

TextGenerator::TextGenerator(InferenceEngine& engine, SamplingConfig config)
    : engine_(engine), config_(std::move(config)) {}

Result<std::vector<TokenId>> TextGenerator::Generate(const std::vector<TokenId>& prompt,
                                                     const std::size_t maxNewTokens, std::mt19937& rng) {
  if (prompt.empty()) {
    return Result<std::vector<TokenId>>::Fail(ErrorCode::InvalidArgument, "generate requires a non-empty prompt");
  }

  if (maxNewTokens == 0) {
    return Result<std::vector<TokenId>>::Ok(prompt);
  }

  const Dimension vocabSize = engine_.GetModelConfig().vocabSize;
  std::vector<TokenId> sequence = prompt;
  Tensor logits = Tensor::Zeros(Shape{1, vocabSize});

  const Status prefillStatus = engine_.Prefill(prompt, logits);
  if (!prefillStatus.IsOk()) {
    return Result<std::vector<TokenId>>::Fail(prefillStatus.GetError().code, prefillStatus.Message());
  }

  if (config_.minFirstTokenProbability > 0.0f) {
    const float confidence = Sampler::MaxTokenProbability(logits);
    if (confidence < config_.minFirstTokenProbability) {
      return Result<std::vector<TokenId>>::Fail(ErrorCode::InvalidArgument,
                                              "generation confidence too low");
    }
  }

  for (std::size_t step = 0; step < maxNewTokens; ++step) {
    const TokenId nextToken = Sampler::Sample(logits, config_, rng);
    sequence.push_back(nextToken);

    if (nextToken == config_.eosTokenId) {
      break;
    }

    if (engine_.SequenceLength() >= engine_.GetModelConfig().maxSeqLen) {
      break;
    }

    const Status decodeStatus = engine_.Decode(nextToken, logits);
    if (!decodeStatus.IsOk()) {
      return Result<std::vector<TokenId>>::Fail(decodeStatus.GetError().code, decodeStatus.Message());
    }
  }

  return Result<std::vector<TokenId>>::Ok(std::move(sequence));
}

} // namespace llm::inference
