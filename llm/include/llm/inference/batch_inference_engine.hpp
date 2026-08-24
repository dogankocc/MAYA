#pragma once

#include <cstddef>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/core/types.hpp"
#include "llm/inference/batch_kv_cache.hpp"
#include "llm/model/transformer.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::inference {

class BatchInferenceEngine {
public:
  BatchInferenceEngine(const model::TransformerModel& model, std::size_t batchSize);

  void Reset();

  void SetActive(const std::vector<bool>& active);

  [[nodiscard]] Status PrefillBatch(const std::vector<std::vector<TokenId>>& prompts, Tensor& logits);

  [[nodiscard]] Status PrefillBatchPadded(const std::vector<std::vector<TokenId>>& prompts, Tensor& logits);

  [[nodiscard]] Status PrefillSlot(std::size_t batchIndex, const std::vector<TokenId>& prompt, Tensor& rowLogits);

  [[nodiscard]] bool UsesPaddedPrefill() const { return usePaddedPrefill_; }

  void SetUsePaddedPrefill(const bool enabled) { usePaddedPrefill_ = enabled; }

  [[nodiscard]] Status DecodeBatch(const std::vector<TokenId>& tokens, Tensor& logits);

  [[nodiscard]] std::size_t BatchSize() const { return batchSize_; }

  [[nodiscard]] const std::vector<bool>& Active() const { return active_; }

  [[nodiscard]] std::size_t SequenceLength(std::size_t batchIndex) const;

  [[nodiscard]] const BatchKvCache& GetCache() const { return cache_; }

  [[nodiscard]] const ModelConfig& GetModelConfig() const { return model_.GetConfig(); }

private:
  [[nodiscard]] Status EmbedTokens(const std::vector<TokenId>& tokens, Tensor& hidden) const;

  [[nodiscard]] Status EmbedBatch(const std::vector<TokenId>& tokens, Tensor& hidden) const;

  [[nodiscard]] Status ForwardHiddenSlot(const Tensor& hidden, std::size_t batchIndex, std::size_t cacheStart,
                                        Tensor& output);

  [[nodiscard]] Status ForwardHiddenBatch(const Tensor& hidden, const std::vector<std::size_t>& cacheStarts,
                                          Tensor& output);

  [[nodiscard]] Status LogitsFromRows(const Tensor& hiddenRows, Tensor& logits) const;

  [[nodiscard]] Status ForwardHiddenPaddedBatch(const Tensor& hidden, std::size_t maxSeqLen,
                                              const std::vector<std::size_t>& seqLens, Tensor& output);

  [[nodiscard]] Status EmbedPaddedBatch(const std::vector<std::vector<TokenId>>& prompts, std::size_t maxSeqLen,
                                      Tensor& hidden, std::vector<std::size_t>& seqLens) const;

  [[nodiscard]] Status LogitsFromPaddedRows(const Tensor& hidden, std::size_t maxSeqLen,
                                            const std::vector<std::size_t>& seqLens, Tensor& logits) const;

  [[nodiscard]] Status LogitsFromSlotLastRow(const Tensor& hidden, Tensor& logitsRow) const;

  const model::TransformerModel& model_;
  std::size_t batchSize_ = 0;
  bool usePaddedPrefill_ = true;
  std::vector<bool> active_;
  BatchKvCache cache_;
};

} // namespace llm::inference
