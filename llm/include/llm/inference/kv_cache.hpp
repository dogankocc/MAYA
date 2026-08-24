#pragma once

#include <cstddef>
#include <vector>

#include "llm/core/config.hpp"
#include "llm/core/status.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::inference {

class KvCache {
public:
  explicit KvCache(const ModelConfig& config);

  void Reset();

  [[nodiscard]] std::size_t Length() const { return length_; }

  [[nodiscard]] std::size_t MaxLength() const { return config_.maxSeqLen; }

  [[nodiscard]] std::size_t NumLayers() const { return keys_.size(); }

  [[nodiscard]] const Tensor& Keys(std::size_t layer) const { return keys_.at(layer); }

  [[nodiscard]] const Tensor& Values(std::size_t layer) const { return values_.at(layer); }

  [[nodiscard]] Status Write(std::size_t layer, std::size_t position, const Tensor& keys, const Tensor& values);

  void SetLength(std::size_t length);

private:
  ModelConfig config_;
  Dimension kvDim_ = 0;
  std::size_t length_ = 0;
  std::vector<Tensor> keys_;
  std::vector<Tensor> values_;
};

} // namespace llm::inference
