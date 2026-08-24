#include "llm/tensor/ops.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "llm/quantization/quantize.hpp"
#include "matmul_internal.hpp"

namespace llm::tensor {

namespace {

[[nodiscard]] bool IsMatrix(const Tensor& tensor) {
  return tensor.Rank() == 2;
}

[[nodiscard]] bool SameShape(const Tensor& a, const Tensor& b) {
  return a.GetShape() == b.GetShape();
}

void ApplyActivation(Tensor& tensor, Scalar (*fn)(Scalar)) {
  for (Index i = 0; i < static_cast<Index>(tensor.Numel()); ++i) {
    tensor[i] = fn(tensor[i]);
  }
}

[[nodiscard]] Scalar GeluScalar(Scalar x) {
  constexpr Scalar kAlpha = 0.044715f;
  constexpr Scalar kSqrtTwoOverPi = 0.7978845608f;
  const Scalar cube = x * x * x;
  const Scalar inner = kSqrtTwoOverPi * (x + kAlpha * cube);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

[[nodiscard]] Scalar SiluScalar(Scalar x) {
  return x / (1.0f + std::exp(-x));
}

void Softmax1D(std::vector<Scalar>& values) {
  if (values.empty()) {
    return;
  }

  const Scalar maxValue = *std::max_element(values.begin(), values.end());
  Scalar sum = 0.0f;

  for (Scalar& value : values) {
    value = std::exp(value - maxValue);
    sum += value;
  }

  if (sum > 0.0f) {
    for (Scalar& value : values) {
      value /= sum;
    }
  }
}

} // namespace

Status MatMul(const Tensor& a, const Tensor& b, Tensor& out) {
  if (!IsMatrix(a) || !IsMatrix(b)) {
    return Status::Fail(ErrorCode::InvalidArgument, "matmul requires rank-2 tensors");
  }

  const Dimension m = a.GetShape()[0];
  const Dimension k = a.GetShape()[1];
  const Dimension bRows = b.GetShape()[0];
  const Dimension n = b.GetShape()[1];

  if (k != bRows) {
    return Status::Fail(ErrorCode::InvalidArgument, "matmul inner dimensions must match");
  }

  if (out.GetShape() != Shape{m, n}) {
    out = Tensor::Zeros(Shape{m, n});
  }

#if defined(__AVX2__) || defined(_M_AVX2)
  detail::MatMulAvx2(a.Data(), b.Data(), out.Data(), m, k, n);
#else
  detail::MatMulScalar(a.Data(), b.Data(), out.Data(), m, k, n);
#endif

  return Status::Ok();
}

Status MatMulQuantized(const Tensor& input, const quantization::QuantizedTensor& weight, Tensor& out) {
  if (!IsMatrix(input) || weight.shape.Rank() != 2) {
    return Status::Fail(ErrorCode::InvalidArgument, "quantized matmul requires rank-2 tensors");
  }

  const Dimension m = input.GetShape()[0];
  const Dimension k = input.GetShape()[1];
  const Dimension n = weight.shape[0];

  if (k != weight.shape[1]) {
    return Status::Fail(ErrorCode::InvalidArgument, "quantized matmul inner dimensions must match");
  }

  if (out.GetShape() != Shape{m, n}) {
    out = Tensor::Zeros(Shape{m, n});
  }

#if defined(__AVX2__) || defined(_M_AVX2)
  detail::MatMulQuantizedAvx2(input.Data(), weight.data.data(), weight.scale, out.Data(), m, k, n);
#else
  detail::MatMulQuantizedScalar(input.Data(), weight.data.data(), weight.scale, out.Data(), m, k, n);
#endif

  return Status::Ok();
}

Status Add(const Tensor& a, const Tensor& b, Tensor& out) {
  if (!SameShape(a, b)) {
    return Status::Fail(ErrorCode::InvalidArgument, "add requires identical shapes");
  }

  if (out.GetShape() != a.GetShape()) {
    out = Tensor::Zeros(a.GetShape());
  }

  for (Index i = 0; i < static_cast<Index>(a.Numel()); ++i) {
    out[i] = a[i] + b[i];
  }

  return Status::Ok();
}

Status Sub(const Tensor& a, const Tensor& b, Tensor& out) {
  if (!SameShape(a, b)) {
    return Status::Fail(ErrorCode::InvalidArgument, "sub requires identical shapes");
  }

  if (out.GetShape() != a.GetShape()) {
    out = Tensor::Zeros(a.GetShape());
  }

  for (Index i = 0; i < static_cast<Index>(a.Numel()); ++i) {
    out[i] = a[i] - b[i];
  }

  return Status::Ok();
}

Status Mul(const Tensor& a, const Tensor& b, Tensor& out) {
  if (!SameShape(a, b)) {
    return Status::Fail(ErrorCode::InvalidArgument, "mul requires identical shapes");
  }

  if (out.GetShape() != a.GetShape()) {
    out = Tensor::Zeros(a.GetShape());
  }

  for (Index i = 0; i < static_cast<Index>(a.Numel()); ++i) {
    out[i] = a[i] * b[i];
  }

  return Status::Ok();
}

Status Scale(const Tensor& input, Scalar factor, Tensor& out) {
  if (out.GetShape() != input.GetShape()) {
    out = Tensor::Zeros(input.GetShape());
  }

  for (Index i = 0; i < static_cast<Index>(input.Numel()); ++i) {
    out[i] = input[i] * factor;
  }

  return Status::Ok();
}

void Relu(Tensor& tensor) {
  ApplyActivation(tensor, [](Scalar x) { return x > 0.0f ? x : 0.0f; });
}

void Gelu(Tensor& tensor) {
  ApplyActivation(tensor, GeluScalar);
}

void Silu(Tensor& tensor) {
  ApplyActivation(tensor, SiluScalar);
}

Status Softmax(Tensor& tensor, Dimension axis) {
  if (tensor.Rank() == 0) {
    return Status::Fail(ErrorCode::InvalidArgument, "softmax requires at least rank-1 tensor");
  }

  if (axis >= tensor.Rank()) {
    return Status::Fail(ErrorCode::OutOfRange, "softmax axis out of range");
  }

  if (tensor.Rank() == 1) {
    if (axis != 0) {
      return Status::Fail(ErrorCode::OutOfRange, "softmax axis out of range");
    }

    std::vector<Scalar> values(tensor.Data(), tensor.Data() + tensor.Numel());
    Softmax1D(values);
    std::copy(values.begin(), values.end(), tensor.Data());
    return Status::Ok();
  }

  if (tensor.Rank() == 2) {
    const Dimension rows = tensor.GetShape()[0];
    const Dimension cols = tensor.GetShape()[1];

    if (axis == 1) {
      std::vector<Scalar> row(cols);
      for (Dimension rowIndex = 0; rowIndex < rows; ++rowIndex) {
        for (Dimension col = 0; col < cols; ++col) {
          row[col] = tensor.At({static_cast<Index>(rowIndex), static_cast<Index>(col)});
        }
        Softmax1D(row);
        for (Dimension col = 0; col < cols; ++col) {
          tensor.At({static_cast<Index>(rowIndex), static_cast<Index>(col)}) = row[col];
        }
      }
      return Status::Ok();
    }

    if (axis == 0) {
      std::vector<Scalar> column(rows);
      for (Dimension col = 0; col < cols; ++col) {
        for (Dimension rowIndex = 0; rowIndex < rows; ++rowIndex) {
          column[rowIndex] = tensor.At({static_cast<Index>(rowIndex), static_cast<Index>(col)});
        }
        Softmax1D(column);
        for (Dimension rowIndex = 0; rowIndex < rows; ++rowIndex) {
          tensor.At({static_cast<Index>(rowIndex), static_cast<Index>(col)}) = column[rowIndex];
        }
      }
      return Status::Ok();
    }
  }

  return Status::Fail(ErrorCode::NotImplemented, "softmax supports rank-1 and rank-2 tensors");
}

} // namespace llm::tensor
