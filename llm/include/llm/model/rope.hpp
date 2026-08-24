#pragma once

#include <cstddef>
#include <vector>

#include "llm/core/types.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::model {

class RopeCache {
public:
  void Build(std::size_t headDim, std::size_t maxSeqLen, float theta);

  [[nodiscard]] Scalar Cos(std::size_t position, std::size_t dimPair) const;

  [[nodiscard]] Scalar Sin(std::size_t position, std::size_t dimPair) const;

  [[nodiscard]] std::size_t HeadDim() const { return headDim_; }

  [[nodiscard]] std::size_t MaxSeqLen() const { return maxSeqLen_; }

private:
  std::size_t headDim_ = 0;
  std::size_t maxSeqLen_ = 0;
  std::vector<Scalar> cos_;
  std::vector<Scalar> sin_;
};

void ApplyRope(Tensor& tensor, const RopeCache& cache, std::size_t numHeads, std::size_t headDim,
               std::size_t positionOffset = 0);

} // namespace llm::model
