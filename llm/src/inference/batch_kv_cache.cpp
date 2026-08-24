#include "llm/inference/batch_kv_cache.hpp"

namespace llm::inference {

BatchKvCache::BatchKvCache(const ModelConfig& config, const std::size_t batchSize)
    : config_(config), batchSize_(batchSize), lengths_(batchSize, 0) {
  const Dimension headDim = config.hiddenDim / config.numHeads;
  kvDim_ = config.numKvHeads * headDim;

  keys_.reserve(config.numLayers);
  values_.reserve(config.numLayers);

  for (std::size_t layer = 0; layer < config.numLayers; ++layer) {
    keys_.push_back(Tensor::Zeros(Shape{batchSize_, config.maxSeqLen, kvDim_}));
    values_.push_back(Tensor::Zeros(Shape{batchSize_, config.maxSeqLen, kvDim_}));
  }
}

void BatchKvCache::Reset() {
  std::fill(lengths_.begin(), lengths_.end(), 0);
}

std::size_t BatchKvCache::Length(const std::size_t batchIndex) const {
  return lengths_.at(batchIndex);
}

void BatchKvCache::SetLength(const std::size_t batchIndex, const std::size_t length) {
  lengths_.at(batchIndex) = length;
}

const Tensor& BatchKvCache::Keys(const std::size_t layer, const std::size_t /*batchIndex*/) const {
  return keys_.at(layer);
}

const Tensor& BatchKvCache::Values(const std::size_t layer, const std::size_t /*batchIndex*/) const {
  return values_.at(layer);
}

Status BatchKvCache::Write(const std::size_t layer, const std::size_t batchIndex, const std::size_t position,
                           const Tensor& keys, const Tensor& values) {
  if (layer >= keys_.size()) {
    return Status::Fail(ErrorCode::OutOfRange, "batch kv cache layer out of range");
  }

  if (batchIndex >= batchSize_) {
    return Status::Fail(ErrorCode::OutOfRange, "batch kv cache batch index out of range");
  }

  if (keys.Rank() != 2 || values.Rank() != 2) {
    return Status::Fail(ErrorCode::InvalidArgument, "batch kv write expects rank-2 tensors");
  }

  const Dimension seqLen = keys.GetShape()[0];
  if (values.GetShape()[0] != seqLen || keys.GetShape()[1] != kvDim_ || values.GetShape()[1] != kvDim_) {
    return Status::Fail(ErrorCode::InvalidArgument, "batch kv tensor shape mismatch");
  }

  if (position + seqLen > config_.maxSeqLen) {
    return Status::Fail(ErrorCode::OutOfRange, "batch kv cache exceeds max_seq_len");
  }

  for (Dimension row = 0; row < seqLen; ++row) {
    for (Dimension col = 0; col < kvDim_; ++col) {
      keys_[layer].At({static_cast<Index>(batchIndex), static_cast<Index>(position + row), static_cast<Index>(col)}) =
          keys.At({static_cast<Index>(row), static_cast<Index>(col)});
      values_[layer].At({static_cast<Index>(batchIndex), static_cast<Index>(position + row), static_cast<Index>(col)}) =
          values.At({static_cast<Index>(row), static_cast<Index>(col)});
    }
  }

  return Status::Ok();
}

} // namespace llm::inference
