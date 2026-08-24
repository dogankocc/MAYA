#include "llm/model/transformer_block.hpp"

#include "llm/model/params.hpp"
#include "llm/model/rms_norm.hpp"
#include "llm/tensor/ops.hpp"

namespace llm::model {

TransformerBlock::TransformerBlock(const ModelConfig& config)
    : config_(config),
      attnNormWeight_(Tensor::Ones(Shape{config.hiddenDim})),
      ffnNormWeight_(Tensor::Ones(Shape{config.hiddenDim})),
      attention_(config),
      ffn_(config) {}

void TransformerBlock::ResetParameters(std::mt19937& rng) {
  attnNormWeight_.Fill(1.0f);
  ffnNormWeight_.Fill(1.0f);
  attention_.ResetParameters(rng);
  ffn_.ResetParameters(rng);
}

Status TransformerBlock::Forward(const Tensor& input, const RopeCache& rope, Tensor& output) const {
  Tensor attnNormed;
  Tensor attnOutput;
  Tensor residual;
  Tensor ffnNormed;
  Tensor ffnOutput;

  const Status norm1Status = RmsNorm(input, attnNormWeight_, config_.normEps, attnNormed);
  if (!norm1Status.IsOk()) {
    return norm1Status;
  }

  const Status attnStatus = attention_.Forward(attnNormed, rope, attnOutput);
  if (!attnStatus.IsOk()) {
    return attnStatus;
  }

  const Status residual1Status = tensor::Add(input, attnOutput, residual);
  if (!residual1Status.IsOk()) {
    return residual1Status;
  }

  const Status norm2Status = RmsNorm(residual, ffnNormWeight_, config_.normEps, ffnNormed);
  if (!norm2Status.IsOk()) {
    return norm2Status;
  }

  const Status ffnStatus = ffn_.Forward(ffnNormed, ffnOutput);
  if (!ffnStatus.IsOk()) {
    return ffnStatus;
  }

  return tensor::Add(residual, ffnOutput, output);
}

Status TransformerBlock::ForwardWithCache(const Tensor& input, const RopeCache& rope, inference::KvCache& cache,
                                          const std::size_t layerIndex, const std::size_t cacheStart,
                                          Tensor& output) const {
  Tensor attnNormed;
  Tensor attnOutput;
  Tensor residual;
  Tensor ffnNormed;
  Tensor ffnOutput;

  const Status norm1Status = RmsNorm(input, attnNormWeight_, config_.normEps, attnNormed);
  if (!norm1Status.IsOk()) {
    return norm1Status;
  }

  const Status attnStatus =
      attention_.ForwardWithCache(attnNormed, rope, cache, layerIndex, cacheStart, attnOutput);
  if (!attnStatus.IsOk()) {
    return attnStatus;
  }

  const Status residual1Status = tensor::Add(input, attnOutput, residual);
  if (!residual1Status.IsOk()) {
    return residual1Status;
  }

  const Status norm2Status = RmsNorm(residual, ffnNormWeight_, config_.normEps, ffnNormed);
  if (!norm2Status.IsOk()) {
    return norm2Status;
  }

  const Status ffnStatus = ffn_.Forward(ffnNormed, ffnOutput);
  if (!ffnStatus.IsOk()) {
    return ffnStatus;
  }

  return tensor::Add(residual, ffnOutput, output);
}

Status TransformerBlock::ForwardWithBatchCache(const Tensor& input, const RopeCache& rope,
                                               inference::BatchKvCache& cache, const std::size_t layerIndex,
                                               const std::vector<std::size_t>& cacheStarts,
                                               const std::vector<bool>& active, Tensor& output) const {
  Tensor attnNormed;
  Tensor attnOutput;
  Tensor residual;
  Tensor ffnNormed;
  Tensor ffnOutput;

  const Status norm1Status = RmsNorm(input, attnNormWeight_, config_.normEps, attnNormed);
  if (!norm1Status.IsOk()) {
    return norm1Status;
  }

  const Status attnStatus =
      attention_.ForwardWithBatchCache(attnNormed, rope, cache, layerIndex, cacheStarts, active, attnOutput);
  if (!attnStatus.IsOk()) {
    return attnStatus;
  }

  const Status residual1Status = tensor::Add(input, attnOutput, residual);
  if (!residual1Status.IsOk()) {
    return residual1Status;
  }

  const Status norm2Status = RmsNorm(residual, ffnNormWeight_, config_.normEps, ffnNormed);
  if (!norm2Status.IsOk()) {
    return norm2Status;
  }

  const Status ffnStatus = ffn_.Forward(ffnNormed, ffnOutput);
  if (!ffnStatus.IsOk()) {
    return ffnStatus;
  }

  return tensor::Add(residual, ffnOutput, output);
}

Status TransformerBlock::ForwardWithBatchSlotCache(const Tensor& input, const RopeCache& rope,
                                                   inference::BatchKvCache& cache, const std::size_t layerIndex,
                                                   const std::size_t batchIndex, const std::size_t cacheStart,
                                                   Tensor& output) const {
  Tensor attnNormed;
  Tensor attnOutput;
  Tensor residual;
  Tensor ffnNormed;
  Tensor ffnOutput;

  const Status norm1Status = RmsNorm(input, attnNormWeight_, config_.normEps, attnNormed);
  if (!norm1Status.IsOk()) {
    return norm1Status;
  }

  const Status attnStatus = attention_.ForwardWithBatchSlotCache(attnNormed, rope, cache, layerIndex, batchIndex,
                                                                 cacheStart, attnOutput);
  if (!attnStatus.IsOk()) {
    return attnStatus;
  }

  const Status residual1Status = tensor::Add(input, attnOutput, residual);
  if (!residual1Status.IsOk()) {
    return residual1Status;
  }

  const Status norm2Status = RmsNorm(residual, ffnNormWeight_, config_.normEps, ffnNormed);
  if (!norm2Status.IsOk()) {
    return norm2Status;
  }

  const Status ffnStatus = ffn_.Forward(ffnNormed, ffnOutput);
  if (!ffnStatus.IsOk()) {
    return ffnStatus;
  }

  return tensor::Add(residual, ffnOutput, output);
}

Status TransformerBlock::ForwardWithBatchPaddedCache(const Tensor& input, const RopeCache& rope,
                                                   inference::BatchKvCache& cache, const std::size_t layerIndex,
                                                   const std::size_t maxSeqLen,
                                                   const std::vector<std::size_t>& seqLens,
                                                   const std::vector<bool>& active, Tensor& output) const {
  Tensor attnNormed;
  Tensor attnOutput;
  Tensor residual;
  Tensor ffnNormed;
  Tensor ffnOutput;

  const Status norm1Status = RmsNorm(input, attnNormWeight_, config_.normEps, attnNormed);
  if (!norm1Status.IsOk()) {
    return norm1Status;
  }

  const Status attnStatus = attention_.ForwardWithBatchPaddedCache(attnNormed, rope, cache, layerIndex, maxSeqLen,
                                                                 seqLens, active, attnOutput);
  if (!attnStatus.IsOk()) {
    return attnStatus;
  }

  const Status residual1Status = tensor::Add(input, attnOutput, residual);
  if (!residual1Status.IsOk()) {
    return residual1Status;
  }

  const Status norm2Status = RmsNorm(residual, ffnNormWeight_, config_.normEps, ffnNormed);
  if (!norm2Status.IsOk()) {
    return norm2Status;
  }

  const Status ffnStatus = ffn_.Forward(ffnNormed, ffnOutput);
  if (!ffnStatus.IsOk()) {
    return ffnStatus;
  }

  return tensor::Add(residual, ffnOutput, output);
}

} // namespace llm::model
