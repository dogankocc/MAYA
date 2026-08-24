#include "llm/model/transformer.hpp"

#include "llm/model/params.hpp"
#include "llm/model/rms_norm.hpp"
#include "llm/tensor/ops.hpp"

namespace llm::model {

TransformerModel::TransformerModel(const ModelConfig& config) : config_(config) {
  const std::size_t headDim = config.hiddenDim / config.numHeads;
  rope_.Build(headDim, config.maxSeqLen, config.ropeTheta);

  tokenEmbedding_ = Tensor::Zeros(Shape{config.vocabSize, config.hiddenDim});
  finalNormWeight_ = Tensor::Ones(Shape{config.hiddenDim});
  lmHeadWeight_ = Tensor::Zeros(Shape{config.vocabSize, config.hiddenDim});

  layers_.reserve(config.numLayers);
  for (std::size_t layer = 0; layer < config.numLayers; ++layer) {
    layers_.emplace_back(config);
  }
}

void TransformerModel::ResetParameters(std::mt19937& rng) {
  InitTensorXavier(tokenEmbedding_, rng);
  finalNormWeight_.Fill(1.0f);
  InitTensorXavier(lmHeadWeight_, rng);

  for (TransformerBlock& layer : layers_) {
    layer.ResetParameters(rng);
  }
}

Status TransformerModel::Forward(const std::vector<TokenId>& tokens, Tensor& logits) const {
  if (tokens.empty()) {
    return Status::Fail(ErrorCode::InvalidArgument, "forward requires at least one token");
  }

  if (tokens.size() > config_.maxSeqLen) {
    return Status::Fail(ErrorCode::InvalidArgument, "sequence length exceeds max_seq_len");
  }

  const Dimension seqLen = tokens.size();
  Tensor hidden = Tensor::Zeros(Shape{seqLen, config_.hiddenDim});

  for (Dimension position = 0; position < seqLen; ++position) {
    const TokenId token = tokens[static_cast<std::size_t>(position)];
    if (token >= config_.vocabSize) {
      return Status::Fail(ErrorCode::OutOfRange, "token id exceeds vocab_size");
    }

    for (Dimension dim = 0; dim < config_.hiddenDim; ++dim) {
      hidden.At({static_cast<Index>(position), static_cast<Index>(dim)}) =
          tokenEmbedding_.At({static_cast<Index>(token), static_cast<Index>(dim)});
    }
  }

  Tensor current = std::move(hidden);
  for (const TransformerBlock& layer : layers_) {
    Tensor next = Tensor::Zeros(Shape{seqLen, config_.hiddenDim});
    const Status layerStatus = layer.Forward(current, rope_, next);
    if (!layerStatus.IsOk()) {
      return layerStatus;
    }
    current = std::move(next);
  }

  Tensor normalized;
  const Status normStatus = RmsNorm(current, finalNormWeight_, config_.normEps, normalized);
  if (!normStatus.IsOk()) {
    return normStatus;
  }

  return LmHeadForward(normalized, logits);
}

Status TransformerModel::LoadQuantizedLmHeadWeight(const quantization::QuantizedTensor& source) {
  if (source.shape.Rank() != 2 || source.shape[0] != config_.vocabSize ||
      source.shape[1] != config_.hiddenDim) {
    return Status::Fail(ErrorCode::InvalidArgument, "quantized lm head shape mismatch");
  }

  quantizedLmHeadWeight_ = source;
  useQuantizedLmHead_ = true;
  return Status::Ok();
}

Status TransformerModel::LmHeadForward(const Tensor& normalized, Tensor& logits) const {
  if (quantizedRuntimeEnabled_ && useQuantizedLmHead_) {
    return tensor::MatMulQuantized(normalized, quantizedLmHeadWeight_, logits);
  }

  const Tensor headTransposed = lmHeadWeight_.Transpose2D();
  return tensor::MatMul(normalized, headTransposed, logits);
}

} // namespace llm::model
