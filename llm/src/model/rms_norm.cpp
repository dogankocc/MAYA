#include "llm/model/rms_norm.hpp"

#include <cmath>

namespace llm::model {

Status RmsNorm(const Tensor& input, const Tensor& weight, const Scalar eps, Tensor& output) {
  if (input.Rank() != 2) {
    return Status::Fail(ErrorCode::InvalidArgument, "rms_norm input must be rank-2");
  }

  const Dimension seqLen = input.GetShape()[0];
  const Dimension hiddenDim = input.GetShape()[1];

  if (weight.Rank() != 1 || weight.GetShape()[0] != hiddenDim) {
    return Status::Fail(ErrorCode::InvalidArgument, "rms_norm weight must be [hidden_dim]");
  }

  output = Tensor::Zeros(Shape{seqLen, hiddenDim});

  for (Dimension row = 0; row < seqLen; ++row) {
    Scalar sumSquares = 0.0f;
    for (Dimension col = 0; col < hiddenDim; ++col) {
      const Scalar value = input.At({static_cast<Index>(row), static_cast<Index>(col)});
      sumSquares += value * value;
    }

    const Scalar rms = std::sqrt(sumSquares / static_cast<Scalar>(hiddenDim) + eps);

    for (Dimension col = 0; col < hiddenDim; ++col) {
      const Scalar value = input.At({static_cast<Index>(row), static_cast<Index>(col)});
      output.At({static_cast<Index>(row), static_cast<Index>(col)}) = (value / rms) * weight[col];
    }
  }

  return Status::Ok();
}

} // namespace llm::model
