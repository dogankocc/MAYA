#pragma once

#include <cstddef>
#include <vector>

#include "llm/core/config.hpp"
#include "llm/core/status.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::inference {

class BatchKvCache {
public:
  BatchKvCache(const ModelConfig& config, std::size_t batchSize);

  void Reset();

  [[nodiscard]] std::size_t BatchSize() const { return batchSize_; }

  [[nodiscard]] std::size_t Length(std::size_t batchIndex) const;

  [[nodiscard]] std::size_t MaxLength() const { return config_.maxSeqLen; }

  [[nodiscard]] std::size_t NumLayers() const { return keys_.size(); }

  [[nodiscard]] const Tensor& Keys(std::size_t layer, std::size_t batchIndex) const;

  [[nodiscard]] const Tensor& Values(std::size_t layer, std::size_t batchIndex) const;

  [[nodiscard]] Status Write(std::size_t layer, std::size_t batchIndex, std::size_t position, const Tensor& keys,
                             const Tensor& values);

  void SetLength(std::size_t batchIndex, std::size_t length);

private:
  ModelConfig config_;
  std::size_t batchSize_ = 0;
  Dimension kvDim_ = 0;
  std::vector<std::size_t> lengths_;
  std::vector<Tensor> keys_;
  std::vector<Tensor> values_;
};

} // namespace llm::inference
