#pragma once

#include <cstdint>
#include <random>

#include "llm/core/status.hpp"
#include "llm/core/types.hpp"
#include "llm/quantization/quantize.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::model {

class Linear {
public:
  Linear() = default;

  Linear(Dimension inputDim, Dimension outputDim, bool useBias = false);

  void ResetParameters(std::mt19937& rng);

  [[nodiscard]] Status Forward(const Tensor& input, Tensor& output) const;

  [[nodiscard]] const Tensor& Weight() const { return weight_; }

  [[nodiscard]] Tensor& WeightMutable() { return weight_; }

  [[nodiscard]] Status LoadWeight(const Tensor& source);

  [[nodiscard]] Status LoadQuantizedWeight(const quantization::QuantizedTensor& source);

  [[nodiscard]] bool UsesQuantizedWeight() const { return useQuantizedWeight_; }

  [[nodiscard]] const quantization::QuantizedTensor* QuantizedWeight() const {
    return useQuantizedWeight_ ? &quantizedWeight_ : nullptr;
  }

  [[nodiscard]] Dimension InputDim() const { return inputDim_; }

  [[nodiscard]] Dimension OutputDim() const { return outputDim_; }

private:
  Dimension inputDim_ = 0;
  Dimension outputDim_ = 0;
  bool useBias_ = false;
  bool useQuantizedWeight_ = false;
  Tensor weight_;
  quantization::QuantizedTensor quantizedWeight_;
  Tensor bias_;
};

} // namespace llm::model
