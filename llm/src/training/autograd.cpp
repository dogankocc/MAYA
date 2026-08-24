#include "llm/training/autograd.hpp"

#include <cmath>

#include "llm/tensor/ops.hpp"

namespace llm::training {

void Accumulate(Tensor& target, const Tensor& source) {
  for (Index i = 0; i < static_cast<Index>(target.Numel()); ++i) {
    target[i] += source[i];
  }
}

Status MatMulBackward(const Tensor& a, const Tensor& b, const Tensor& gradOutput, Tensor& gradA, Tensor& gradB) {
  if (gradOutput.Rank() != 2 || a.Rank() != 2 || b.Rank() != 2) {
    return Status::Fail(ErrorCode::InvalidArgument, "matmul backward expects rank-2 tensors");
  }

  const Tensor bTransposed = b.Transpose2D();
  const Tensor aTransposed = a.Transpose2D();

  if (gradA.GetShape() != a.GetShape()) {
    gradA = Tensor::Zeros(a.GetShape());
  } else {
    gradA.Fill(0.0f);
  }

  if (gradB.GetShape() != b.GetShape()) {
    gradB = Tensor::Zeros(b.GetShape());
  } else {
    gradB.Fill(0.0f);
  }

  const Status gradAStatus = tensor::MatMul(gradOutput, bTransposed, gradA);
  const Status gradBStatus = tensor::MatMul(aTransposed, gradOutput, gradB);
  if (!gradAStatus.IsOk() || !gradBStatus.IsOk()) {
    return Status::Fail(ErrorCode::Internal, "matmul backward failed");
  }

  return Status::Ok();
}

Status AddTo(Tensor& target, const Tensor& source) {
  if (target.GetShape() != source.GetShape()) {
    return Status::Fail(ErrorCode::InvalidArgument, "addto shape mismatch");
  }

  for (Index i = 0; i < static_cast<Index>(target.Numel()); ++i) {
    target[i] += source[i];
  }

  return Status::Ok();
}

Status MulBackward(const Tensor& a, const Tensor& b, const Tensor& gradOutput, Tensor& gradA, Tensor& gradB) {
  if (gradOutput.GetShape() != a.GetShape() || a.GetShape() != b.GetShape()) {
    return Status::Fail(ErrorCode::InvalidArgument, "mul backward shape mismatch");
  }

  gradA = Tensor::Zeros(a.GetShape());
  gradB = Tensor::Zeros(b.GetShape());

  for (Index i = 0; i < static_cast<Index>(gradOutput.Numel()); ++i) {
    gradA[i] = gradOutput[i] * b[i];
    gradB[i] = gradOutput[i] * a[i];
  }

  return Status::Ok();
}

void SiluBackward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) {
  gradInput = Tensor::Zeros(input.GetShape());

  for (Index i = 0; i < static_cast<Index>(input.Numel()); ++i) {
    const Scalar x = input[i];
    const Scalar sigmoid = 1.0f / (1.0f + std::exp(-x));
    const Scalar derivative = sigmoid * (1.0f + x * (1.0f - sigmoid));
    gradInput[i] = gradOutput[i] * derivative;
    (void)output;
  }
}

Status RmsNormBackward(const Tensor& input, const Tensor& weight, const Tensor& output, const Tensor& gradOutput,
                       const Scalar eps, Tensor& gradInput, Tensor& gradWeight) {
  if (input.Rank() != 2 || gradOutput.Rank() != 2) {
    return Status::Fail(ErrorCode::InvalidArgument, "rmsnorm backward expects rank-2 tensors");
  }

  const Dimension seqLen = input.GetShape()[0];
  const Dimension hiddenDim = input.GetShape()[1];

  gradInput = Tensor::Zeros(input.GetShape());
  gradWeight = Tensor::Zeros(weight.GetShape());

  for (Dimension row = 0; row < seqLen; ++row) {
    Scalar sumSquares = 0.0f;
    for (Dimension dim = 0; dim < hiddenDim; ++dim) {
      const Scalar value = input.At({static_cast<Index>(row), static_cast<Index>(dim)});
      sumSquares += value * value;
    }

    const Scalar rms = std::sqrt(sumSquares / static_cast<Scalar>(hiddenDim) + eps);
    const Scalar invRms = 1.0f / rms;

    Scalar gradDot = 0.0f;
    for (Dimension dim = 0; dim < hiddenDim; ++dim) {
      const Scalar gradOut = gradOutput.At({static_cast<Index>(row), static_cast<Index>(dim)});
      const Scalar w = weight[static_cast<Index>(dim)];
      gradDot += gradOut * w * input.At({static_cast<Index>(row), static_cast<Index>(dim)});
    }

    for (Dimension dim = 0; dim < hiddenDim; ++dim) {
      const Scalar x = input.At({static_cast<Index>(row), static_cast<Index>(dim)});
      const Scalar w = weight[static_cast<Index>(dim)];
      const Scalar gradOut = gradOutput.At({static_cast<Index>(row), static_cast<Index>(dim)});

      gradWeight[static_cast<Index>(dim)] += gradOut * x * invRms;

      const Scalar gradX =
          gradOut * w * invRms - (invRms * invRms * invRms / static_cast<Scalar>(hiddenDim)) * x * gradDot;
      gradInput.At({static_cast<Index>(row), static_cast<Index>(dim)}) = gradX;
      (void)output;
    }
  }

  return Status::Ok();
}

Status LinearBackward(const Tensor& input, const Tensor& weight, const Tensor& gradOutput, Tensor& gradInput,
                      Tensor& gradWeight) {
  const Tensor weightTransposed = weight.Transpose2D();
  Tensor gradInputTmp;
  Tensor gradWeightTmp;
  const Status status = MatMulBackward(input, weightTransposed, gradOutput, gradInputTmp, gradWeightTmp);
  if (!status.IsOk()) {
    return status;
  }

  gradInput = std::move(gradInputTmp);
  gradWeight = std::move(gradWeightTmp);
  return Status::Ok();
}

} // namespace llm::training
