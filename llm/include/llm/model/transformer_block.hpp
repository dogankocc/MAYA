#pragma once

#include <random>

#include "llm/core/config.hpp"
#include "llm/core/status.hpp"
#include "llm/inference/batch_kv_cache.hpp"
#include "llm/inference/kv_cache.hpp"
#include "llm/model/attention.hpp"
#include "llm/model/ffn.hpp"
#include "llm/model/rope.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::model {

class TransformerBlock {
public:
  explicit TransformerBlock(const ModelConfig& config);

  void ResetParameters(std::mt19937& rng);

  [[nodiscard]] Status Forward(const Tensor& input, const RopeCache& rope, Tensor& output) const;

  [[nodiscard]] Status ForwardWithCache(const Tensor& input, const RopeCache& rope, inference::KvCache& cache,
                                        std::size_t layerIndex, std::size_t cacheStart, Tensor& output) const;

  [[nodiscard]] Status ForwardWithBatchCache(const Tensor& input, const RopeCache& rope,
                                             inference::BatchKvCache& cache, std::size_t layerIndex,
                                             const std::vector<std::size_t>& cacheStarts,
                                             const std::vector<bool>& active, Tensor& output) const;

  [[nodiscard]] Status ForwardWithBatchSlotCache(const Tensor& input, const RopeCache& rope,
                                                 inference::BatchKvCache& cache, std::size_t layerIndex,
                                                 std::size_t batchIndex, std::size_t cacheStart, Tensor& output) const;

  [[nodiscard]] Status ForwardWithBatchPaddedCache(const Tensor& input, const RopeCache& rope,
                                                   inference::BatchKvCache& cache, std::size_t layerIndex,
                                                   std::size_t maxSeqLen, const std::vector<std::size_t>& seqLens,
                                                   const std::vector<bool>& active, Tensor& output) const;

  [[nodiscard]] const Tensor& AttnNormWeight() const { return attnNormWeight_; }

  [[nodiscard]] const Tensor& FfnNormWeight() const { return ffnNormWeight_; }

  [[nodiscard]] Tensor& AttnNormWeightMutable() { return attnNormWeight_; }

  [[nodiscard]] Tensor& FfnNormWeightMutable() { return ffnNormWeight_; }

  [[nodiscard]] MultiHeadAttention& AttentionModule() { return attention_; }

  [[nodiscard]] SwiGluFfn& FfnModule() { return ffn_; }

  [[nodiscard]] const MultiHeadAttention& AttentionModule() const { return attention_; }

  [[nodiscard]] const SwiGluFfn& FfnModule() const { return ffn_; }

private:
  ModelConfig config_;
  Tensor attnNormWeight_;
  Tensor ffnNormWeight_;
  MultiHeadAttention attention_;
  SwiGluFfn ffn_;
};

} // namespace llm::model
