#include "llm/model/linear.hpp"

#include "llm/model/params.hpp"
#include "llm/tensor/ops.hpp"

namespace llm::model {

Linear::Linear(const Dimension inputDim, const Dimension outputDim, const bool useBias)
    : inputDim_(inputDim),
      outputDim_(outputDim),
      useBias_(useBias),
      weight_(Tensor::Zeros(Shape{outputDim, inputDim})),
      bias_(useBias ? Tensor::Zeros(Shape{outputDim}) : Tensor{}) {}

void Linear::ResetParameters(std::mt19937& rng) {
  InitTensorXavier(weight_, rng);
  if (useBias_) {
    bias_.Fill(0.0f);
  }
}

Status Linear::Forward(const Tensor& input, Tensor& output) const {
  if (input.Rank() != 2 || input.GetShape()[1] != inputDim_) {
    return Status::Fail(ErrorCode::InvalidArgument, "linear input must be [seq, input_dim]");
  }

  if (useQuantizedWeight_) {
    return tensor::MatMulQuantized(input, quantizedWeight_, output);
  }

  const Tensor weightTransposed = weight_.Transpose2D();
  const Status matmulStatus = tensor::MatMul(input, weightTransposed, output);
  if (!matmulStatus.IsOk()) {
    return matmulStatus;
  }

  if (!useBias_) {
    return Status::Ok();
  }

  const Dimension seqLen = input.GetShape()[0];
  for (Dimension row = 0; row < seqLen; ++row) {
    for (Dimension col = 0; col < outputDim_; ++col) {
      output.At({static_cast<Index>(row), static_cast<Index>(col)}) += bias_[col];
    }
  }

  return Status::Ok();
}

Status Linear::LoadWeight(const Tensor& source) {
  if (source.GetShape() != weight_.GetShape()) {
    return Status::Fail(ErrorCode::InvalidArgument, "linear weight shape mismatch");
  }

  for (Index i = 0; i < static_cast<Index>(weight_.Numel()); ++i) {
    weight_[i] = source[i];
  }

  useQuantizedWeight_ = false;
  return Status::Ok();
}

Status Linear::LoadQuantizedWeight(const quantization::QuantizedTensor& source) {
  if (source.shape.Rank() != 2 || source.shape[0] != outputDim_ || source.shape[1] != inputDim_) {
    return Status::Fail(ErrorCode::InvalidArgument, "quantized linear weight shape mismatch");
  }

  quantizedWeight_ = source;
  useQuantizedWeight_ = true;
  return Status::Ok();
}

} // namespace llm::model
