#include "llm/model/params.hpp"

#include <cmath>

namespace llm::model {

void InitTensorXavier(Tensor& tensor, std::mt19937& rng) {
  if (tensor.Numel() == 0) {
    return;
  }

  const Dimension fanIn = tensor.GetShape().Rank() >= 2 ? tensor.GetShape()[1] : 1;
  const Dimension fanOut = tensor.GetShape()[0];
  const float limit = std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));

  std::uniform_real_distribution<float> distribution(-limit, limit);
  for (Index i = 0; i < static_cast<Index>(tensor.Numel()); ++i) {
    tensor[i] = distribution(rng);
  }
}

void InitTensorZeros(Tensor& tensor) {
  tensor.Fill(0.0f);
}

} // namespace llm::model
