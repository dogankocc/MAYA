#include "llm/model/rope.hpp"

#include <cmath>

namespace llm::model {

void RopeCache::Build(const std::size_t headDim, const std::size_t maxSeqLen, const float theta) {
  headDim_ = headDim;
  maxSeqLen_ = maxSeqLen;

  const std::size_t pairs = headDim / 2;
  cos_.assign(maxSeqLen * pairs, 0.0f);
  sin_.assign(maxSeqLen * pairs, 0.0f);

  for (std::size_t position = 0; position < maxSeqLen; ++position) {
    for (std::size_t pair = 0; pair < pairs; ++pair) {
      const float exponent = static_cast<float>(2 * pair) / static_cast<float>(headDim);
      const float frequency = std::pow(theta, -exponent);
      const float angle = static_cast<float>(position) * frequency;
      const std::size_t index = position * pairs + pair;
      cos_[index] = std::cos(angle);
      sin_[index] = std::sin(angle);
    }
  }
}

Scalar RopeCache::Cos(const std::size_t position, const std::size_t dimPair) const {
  return cos_[position * (headDim_ / 2) + dimPair];
}

Scalar RopeCache::Sin(const std::size_t position, const std::size_t dimPair) const {
  return sin_[position * (headDim_ / 2) + dimPair];
}

void ApplyRope(Tensor& tensor, const RopeCache& cache, const std::size_t numHeads, const std::size_t headDim,
               const std::size_t positionOffset) {
  if (tensor.Rank() != 2) {
    return;
  }

  const Dimension seqLen = tensor.GetShape()[0];
  const Dimension rowWidth = tensor.GetShape()[1];

  for (Dimension seq = 0; seq < seqLen; ++seq) {
    for (std::size_t head = 0; head < numHeads; ++head) {
      for (std::size_t dim = 0; dim < headDim; dim += 2) {
        const Index base = static_cast<Index>(seq) * rowWidth + static_cast<Index>(head * headDim + dim);
        const Scalar x0 = tensor[base];
        const Scalar x1 = tensor[base + 1];
        const Scalar cosValue = cache.Cos(positionOffset + static_cast<std::size_t>(seq), dim / 2);
        const Scalar sinValue = cache.Sin(positionOffset + static_cast<std::size_t>(seq), dim / 2);

        tensor[base] = x0 * cosValue - x1 * sinValue;
        tensor[base + 1] = x0 * sinValue + x1 * cosValue;
      }
    }
  }
}

} // namespace llm::model
