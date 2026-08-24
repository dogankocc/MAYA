#include "llm/inference/sampling/batch_generator.hpp"

#include <algorithm>

#include "llm/tensor/tensor.hpp"

namespace llm::inference {

BatchTextGenerator::BatchTextGenerator(BatchInferenceEngine& engine, SamplingConfig config)
    : engine_(engine), config_(std::move(config)) {}

Result<std::vector<std::vector<TokenId>>> BatchTextGenerator::GenerateBatch(
    const std::vector<std::vector<TokenId>>& prompts, const std::size_t maxNewTokens, std::mt19937& rng) {
  if (prompts.size() != engine_.BatchSize()) {
    return Result<std::vector<std::vector<TokenId>>>::Fail(ErrorCode::InvalidArgument,
                                                           "prompt batch size mismatch");
  }

  if (maxNewTokens == 0) {
    return Result<std::vector<std::vector<TokenId>>>::Ok(prompts);
  }

  const Dimension vocabSize = engine_.GetModelConfig().vocabSize;
  Tensor logits = Tensor::Zeros(Shape{engine_.BatchSize(), vocabSize});
  const Status prefillStatus = engine_.PrefillBatch(prompts, logits);
  if (!prefillStatus.IsOk()) {
    return Result<std::vector<std::vector<TokenId>>>::Fail(prefillStatus.GetError().code, prefillStatus.Message());
  }

  std::vector<std::vector<TokenId>> sequences = prompts;
  std::vector<bool> active(engine_.BatchSize(), true);

  for (std::size_t step = 0; step < maxNewTokens; ++step) {
    std::vector<TokenId> nextTokens(engine_.BatchSize(), 0);

    for (std::size_t batchIndex = 0; batchIndex < engine_.BatchSize(); ++batchIndex) {
      if (!active[batchIndex]) {
        continue;
      }

      nextTokens[batchIndex] = Sampler::SampleRow(logits, batchIndex, config_, rng);
      sequences[batchIndex].push_back(nextTokens[batchIndex]);

      if (nextTokens[batchIndex] == config_.eosTokenId) {
        active[batchIndex] = false;
      }

      if (engine_.SequenceLength(batchIndex) >= engine_.GetModelConfig().maxSeqLen) {
        active[batchIndex] = false;
      }
    }

    if (std::none_of(active.begin(), active.end(), [](const bool value) { return value; })) {
      break;
    }

    engine_.SetActive(active);
    const Status decodeStatus = engine_.DecodeBatch(nextTokens, logits);
    if (!decodeStatus.IsOk()) {
      return Result<std::vector<std::vector<TokenId>>>::Fail(decodeStatus.GetError().code, decodeStatus.Message());
    }
  }

  return Result<std::vector<std::vector<TokenId>>>::Ok(std::move(sequences));
}

} // namespace llm::inference
