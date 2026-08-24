#pragma once

#include <random>

#include "llm/core/config.hpp"
#include "llm/core/status.hpp"
#include "llm/inference/batch_kv_cache.hpp"
#include "llm/inference/kv_cache.hpp"
#include "llm/model/linear.hpp"
#include "llm/model/rope.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::model {

class MultiHeadAttention {
public:
  explicit MultiHeadAttention(const ModelConfig& config);

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

  [[nodiscard]] Linear& QueryProjection() { return query_; }

  [[nodiscard]] Linear& KeyProjection() { return key_; }

  [[nodiscard]] Linear& ValueProjection() { return value_; }

  [[nodiscard]] Linear& OutputProjection() { return output_; }

  [[nodiscard]] const Linear& QueryProjection() const { return query_; }

  [[nodiscard]] const Linear& KeyProjection() const { return key_; }

  [[nodiscard]] const Linear& ValueProjection() const { return value_; }

  [[nodiscard]] const Linear& OutputProjection() const { return output_; }

private:
  ModelConfig config_;
  Linear query_;
  Linear key_;
  Linear value_;
  Linear output_;
};

} // namespace llm::model
