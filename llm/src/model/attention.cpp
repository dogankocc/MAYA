#include "llm/model/attention.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "llm/inference/kv_cache.hpp"
#include "llm/model/rope.hpp"
#include "llm/tensor/ops.hpp"

namespace llm::model {

MultiHeadAttention::MultiHeadAttention(const ModelConfig& config) : config_(config) {
  const Dimension headDim = config.hiddenDim / config.numHeads;
  const Dimension kvDim = config.numKvHeads * headDim;

  query_ = Linear(config.hiddenDim, config.hiddenDim);
  key_ = Linear(config.hiddenDim, kvDim);
  value_ = Linear(config.hiddenDim, kvDim);
  output_ = Linear(config.hiddenDim, config.hiddenDim);
}

void MultiHeadAttention::ResetParameters(std::mt19937& rng) {
  query_.ResetParameters(rng);
  key_.ResetParameters(rng);
  value_.ResetParameters(rng);
  output_.ResetParameters(rng);
}

Status MultiHeadAttention::Forward(const Tensor& input, const RopeCache& rope, Tensor& output) const {
  if (input.Rank() != 2 || input.GetShape()[1] != config_.hiddenDim) {
    return Status::Fail(ErrorCode::InvalidArgument, "attention input must be [seq, hidden_dim]");
  }

  const Dimension seqLen = input.GetShape()[0];
  const Dimension numHeads = config_.numHeads;
  const Dimension numKvHeads = config_.numKvHeads;
  const Dimension headDim = config_.hiddenDim / numHeads;
  const Dimension kvDim = numKvHeads * headDim;
  const Dimension headsPerKv = numHeads / numKvHeads;
  const Scalar scale = 1.0f / std::sqrt(static_cast<Scalar>(headDim));

  Tensor queries;
  Tensor keys;
  Tensor values;
  const Status queryStatus = query_.Forward(input, queries);
  const Status keyStatus = key_.Forward(input, keys);
  const Status valueStatus = value_.Forward(input, values);
  if (!queryStatus.IsOk() || !keyStatus.IsOk() || !valueStatus.IsOk()) {
    return Status::Fail(ErrorCode::Internal, "attention projection failed");
  }

  ApplyRope(queries, rope, numHeads, headDim);
  ApplyRope(keys, rope, numKvHeads, headDim);

  Tensor merged = Tensor::Zeros(Shape{seqLen, config_.hiddenDim});
  const Scalar maskValue = -std::numeric_limits<Scalar>::infinity();

  for (Dimension queryPos = 0; queryPos < seqLen; ++queryPos) {
    for (Dimension head = 0; head < numHeads; ++head) {
      const Dimension kvHead = head / headsPerKv;

      std::vector<Scalar> scores(static_cast<std::size_t>(seqLen), maskValue);
      for (Dimension keyPos = 0; keyPos <= queryPos; ++keyPos) {
        Scalar dot = 0.0f;
        for (Dimension dim = 0; dim < headDim; ++dim) {
          const Scalar qValue =
              queries.At({static_cast<Index>(queryPos), static_cast<Index>(head * headDim + dim)});
          const Scalar kValue =
              keys.At({static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)});
          dot += qValue * kValue;
        }
        scores[static_cast<std::size_t>(keyPos)] = dot * scale;
      }

      Scalar maxScore = scores[0];
      for (Dimension keyPos = 1; keyPos <= queryPos; ++keyPos) {
        maxScore = std::max(maxScore, scores[static_cast<std::size_t>(keyPos)]);
      }

      Scalar sumExp = 0.0f;
      for (Dimension keyPos = 0; keyPos <= queryPos; ++keyPos) {
        scores[static_cast<std::size_t>(keyPos)] = std::exp(scores[static_cast<std::size_t>(keyPos)] - maxScore);
        sumExp += scores[static_cast<std::size_t>(keyPos)];
      }

      for (Dimension dim = 0; dim < headDim; ++dim) {
        Scalar weighted = 0.0f;
        for (Dimension keyPos = 0; keyPos <= queryPos; ++keyPos) {
          const Scalar attnWeight = scores[static_cast<std::size_t>(keyPos)] / sumExp;
          const Scalar vValue =
              values.At({static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)});
          weighted += attnWeight * vValue;
        }
        merged.At({static_cast<Index>(queryPos), static_cast<Index>(head * headDim + dim)}) = weighted;
      }
    }
  }

  return output_.Forward(merged, output);
}

Status MultiHeadAttention::ForwardWithCache(const Tensor& input, const RopeCache& rope, inference::KvCache& cache,
                                            const std::size_t layerIndex, const std::size_t cacheStart,
                                            Tensor& output) const {
  if (input.Rank() != 2 || input.GetShape()[1] != config_.hiddenDim) {
    return Status::Fail(ErrorCode::InvalidArgument, "attention input must be [seq, hidden_dim]");
  }

  const Dimension seqLen = input.GetShape()[0];
  const Dimension numHeads = config_.numHeads;
  const Dimension numKvHeads = config_.numKvHeads;
  const Dimension headDim = config_.hiddenDim / numHeads;
  const Dimension headsPerKv = numHeads / numKvHeads;
  const Scalar scale = 1.0f / std::sqrt(static_cast<Scalar>(headDim));

  Tensor queries;
  Tensor keys;
  Tensor values;
  const Status queryStatus = query_.Forward(input, queries);
  const Status keyStatus = key_.Forward(input, keys);
  const Status valueStatus = value_.Forward(input, values);
  if (!queryStatus.IsOk() || !keyStatus.IsOk() || !valueStatus.IsOk()) {
    return Status::Fail(ErrorCode::Internal, "attention projection failed");
  }

  ApplyRope(queries, rope, numHeads, headDim, cacheStart);
  ApplyRope(keys, rope, numKvHeads, headDim, cacheStart);

  const Status writeStatus = cache.Write(layerIndex, cacheStart, keys, values);
  if (!writeStatus.IsOk()) {
    return writeStatus;
  }

  const std::size_t totalLen = cacheStart + static_cast<std::size_t>(seqLen);
  cache.SetLength(totalLen);

  const Tensor& cachedKeys = cache.Keys(layerIndex);
  const Tensor& cachedValues = cache.Values(layerIndex);

  Tensor merged = Tensor::Zeros(Shape{seqLen, config_.hiddenDim});
  const Scalar maskValue = -std::numeric_limits<Scalar>::infinity();

  for (Dimension localQuery = 0; localQuery < seqLen; ++localQuery) {
    const std::size_t globalQuery = cacheStart + static_cast<std::size_t>(localQuery);

    for (Dimension head = 0; head < numHeads; ++head) {
      const Dimension kvHead = head / headsPerKv;

      std::vector<Scalar> scores(globalQuery + 1, maskValue);
      for (std::size_t keyPos = 0; keyPos <= globalQuery; ++keyPos) {
        Scalar dot = 0.0f;
        for (Dimension dim = 0; dim < headDim; ++dim) {
          const Scalar qValue =
              queries.At({static_cast<Index>(localQuery), static_cast<Index>(head * headDim + dim)});
          const Scalar kValue =
              cachedKeys.At({static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)});
          dot += qValue * kValue;
        }
        scores[keyPos] = dot * scale;
      }

      Scalar maxScore = scores[0];
      for (std::size_t keyPos = 1; keyPos <= globalQuery; ++keyPos) {
        maxScore = std::max(maxScore, scores[keyPos]);
      }

      Scalar sumExp = 0.0f;
      for (std::size_t keyPos = 0; keyPos <= globalQuery; ++keyPos) {
        scores[keyPos] = std::exp(scores[keyPos] - maxScore);
        sumExp += scores[keyPos];
      }

      for (Dimension dim = 0; dim < headDim; ++dim) {
        Scalar weighted = 0.0f;
        for (std::size_t keyPos = 0; keyPos <= globalQuery; ++keyPos) {
          const Scalar attnWeight = scores[keyPos] / sumExp;
          const Scalar vValue =
              cachedValues.At({static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)});
          weighted += attnWeight * vValue;
        }
        merged.At({static_cast<Index>(localQuery), static_cast<Index>(head * headDim + dim)}) = weighted;
      }
    }
  }

  return output_.Forward(merged, output);
}

Status MultiHeadAttention::ForwardWithBatchCache(const Tensor& input, const RopeCache& rope,
                                                 inference::BatchKvCache& cache, const std::size_t layerIndex,
                                                 const std::vector<std::size_t>& cacheStarts,
                                                 const std::vector<bool>& active, Tensor& output) const {
  if (input.Rank() != 2 || input.GetShape()[1] != config_.hiddenDim) {
    return Status::Fail(ErrorCode::InvalidArgument, "batch attention input must be [batch, hidden_dim]");
  }

  const Dimension batchSize = input.GetShape()[0];
  if (cacheStarts.size() != static_cast<std::size_t>(batchSize) ||
      active.size() != static_cast<std::size_t>(batchSize)) {
    return Status::Fail(ErrorCode::InvalidArgument, "cacheStarts/active size must match batch size");
  }

  if (static_cast<std::size_t>(batchSize) != cache.BatchSize()) {
    return Status::Fail(ErrorCode::InvalidArgument, "batch attention batch size mismatch");
  }

  const Dimension numHeads = config_.numHeads;
  const Dimension numKvHeads = config_.numKvHeads;
  const Dimension headDim = config_.hiddenDim / numHeads;
  const Dimension headsPerKv = numHeads / numKvHeads;
  const Scalar scale = 1.0f / std::sqrt(static_cast<Scalar>(headDim));
  const Scalar maskValue = -std::numeric_limits<Scalar>::infinity();

  output = Tensor::Zeros(Shape{batchSize, config_.hiddenDim});

  for (Dimension batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
    if (!active[static_cast<std::size_t>(batchIndex)]) {
      continue;
    }

    const std::size_t cacheStart = cacheStarts[static_cast<std::size_t>(batchIndex)];

    Tensor rowInput = Tensor::Zeros(Shape{1, config_.hiddenDim});
    for (Dimension dim = 0; dim < config_.hiddenDim; ++dim) {
      rowInput.At({0, static_cast<Index>(dim)}) =
          input.At({static_cast<Index>(batchIndex), static_cast<Index>(dim)});
    }

    Tensor queries;
    Tensor keys;
    Tensor values;
    const Status queryStatus = query_.Forward(rowInput, queries);
    const Status keyStatus = key_.Forward(rowInput, keys);
    const Status valueStatus = value_.Forward(rowInput, values);
    if (!queryStatus.IsOk() || !keyStatus.IsOk() || !valueStatus.IsOk()) {
      return Status::Fail(ErrorCode::Internal, "batch attention projection failed");
    }

    ApplyRope(queries, rope, numHeads, headDim, cacheStart);
    ApplyRope(keys, rope, numKvHeads, headDim, cacheStart);

    const Status writeStatus =
        cache.Write(layerIndex, static_cast<std::size_t>(batchIndex), cacheStart, keys, values);
    if (!writeStatus.IsOk()) {
      return writeStatus;
    }

    const std::size_t totalLen = cacheStart + 1;
    cache.SetLength(static_cast<std::size_t>(batchIndex), totalLen);

    const Tensor& cachedKeys = cache.Keys(layerIndex, static_cast<std::size_t>(batchIndex));
    const Tensor& cachedValues = cache.Values(layerIndex, static_cast<std::size_t>(batchIndex));

    for (Dimension head = 0; head < numHeads; ++head) {
      const Dimension kvHead = head / headsPerKv;

      std::vector<Scalar> scores(totalLen, maskValue);
      for (std::size_t keyPos = 0; keyPos < totalLen; ++keyPos) {
        Scalar dot = 0.0f;
        for (Dimension dim = 0; dim < headDim; ++dim) {
          const Scalar qValue = queries.At({0, static_cast<Index>(head * headDim + dim)});
          const Scalar kValue =
              cachedKeys.At({static_cast<Index>(batchIndex), static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)});
          dot += qValue * kValue;
        }
        scores[keyPos] = dot * scale;
      }

      Scalar maxScore = scores[0];
      for (std::size_t keyPos = 1; keyPos < totalLen; ++keyPos) {
        maxScore = std::max(maxScore, scores[keyPos]);
      }

      Scalar sumExp = 0.0f;
      for (std::size_t keyPos = 0; keyPos < totalLen; ++keyPos) {
        scores[keyPos] = std::exp(scores[keyPos] - maxScore);
        sumExp += scores[keyPos];
      }

      for (Dimension dim = 0; dim < headDim; ++dim) {
        Scalar weighted = 0.0f;
        for (std::size_t keyPos = 0; keyPos < totalLen; ++keyPos) {
          const Scalar attnWeight = scores[keyPos] / sumExp;
          const Scalar vValue = cachedValues.At(
              {static_cast<Index>(batchIndex), static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)});
          weighted += attnWeight * vValue;
        }
        output.At({static_cast<Index>(batchIndex), static_cast<Index>(head * headDim + dim)}) = weighted;
      }
    }
  }

  Tensor projected;
  const Status outputStatus = output_.Forward(output, projected);
  if (!outputStatus.IsOk()) {
    return outputStatus;
  }

  output = std::move(projected);
  return Status::Ok();
}

Status MultiHeadAttention::ForwardWithBatchSlotCache(const Tensor& input, const RopeCache& rope,
                                                     inference::BatchKvCache& cache, const std::size_t layerIndex,
                                                     const std::size_t batchIndex, const std::size_t cacheStart,
                                                     Tensor& output) const {
  if (input.Rank() != 2 || input.GetShape()[1] != config_.hiddenDim) {
    return Status::Fail(ErrorCode::InvalidArgument, "batch slot attention input must be [seq, hidden_dim]");
  }

  const Dimension seqLen = input.GetShape()[0];
  const Dimension numHeads = config_.numHeads;
  const Dimension numKvHeads = config_.numKvHeads;
  const Dimension headDim = config_.hiddenDim / numHeads;
  const Dimension headsPerKv = numHeads / numKvHeads;
  const Scalar scale = 1.0f / std::sqrt(static_cast<Scalar>(headDim));

  Tensor queries;
  Tensor keys;
  Tensor values;
  const Status queryStatus = query_.Forward(input, queries);
  const Status keyStatus = key_.Forward(input, keys);
  const Status valueStatus = value_.Forward(input, values);
  if (!queryStatus.IsOk() || !keyStatus.IsOk() || !valueStatus.IsOk()) {
    return Status::Fail(ErrorCode::Internal, "batch slot attention projection failed");
  }

  ApplyRope(queries, rope, numHeads, headDim, cacheStart);
  ApplyRope(keys, rope, numKvHeads, headDim, cacheStart);

  const Status writeStatus = cache.Write(layerIndex, batchIndex, cacheStart, keys, values);
  if (!writeStatus.IsOk()) {
    return writeStatus;
  }

  const std::size_t totalLen = cacheStart + static_cast<std::size_t>(seqLen);
  cache.SetLength(batchIndex, totalLen);

  const Tensor& cachedKeys = cache.Keys(layerIndex, batchIndex);
  const Tensor& cachedValues = cache.Values(layerIndex, batchIndex);

  Tensor merged = Tensor::Zeros(Shape{seqLen, config_.hiddenDim});
  const Scalar maskValue = -std::numeric_limits<Scalar>::infinity();

  for (Dimension localQuery = 0; localQuery < seqLen; ++localQuery) {
    const std::size_t globalQuery = cacheStart + static_cast<std::size_t>(localQuery);

    for (Dimension head = 0; head < numHeads; ++head) {
      const Dimension kvHead = head / headsPerKv;

      std::vector<Scalar> scores(globalQuery + 1, maskValue);
      for (std::size_t keyPos = 0; keyPos <= globalQuery; ++keyPos) {
        Scalar dot = 0.0f;
        for (Dimension dim = 0; dim < headDim; ++dim) {
          const Scalar qValue =
              queries.At({static_cast<Index>(localQuery), static_cast<Index>(head * headDim + dim)});
          const Scalar kValue = cachedKeys.At({static_cast<Index>(batchIndex), static_cast<Index>(keyPos),
                                               static_cast<Index>(kvHead * headDim + dim)});
          dot += qValue * kValue;
        }
        scores[keyPos] = dot * scale;
      }

      Scalar maxScore = scores[0];
      for (std::size_t keyPos = 1; keyPos <= globalQuery; ++keyPos) {
        maxScore = std::max(maxScore, scores[keyPos]);
      }

      Scalar sumExp = 0.0f;
      for (std::size_t keyPos = 0; keyPos <= globalQuery; ++keyPos) {
        scores[keyPos] = std::exp(scores[keyPos] - maxScore);
        sumExp += scores[keyPos];
      }

      for (Dimension dim = 0; dim < headDim; ++dim) {
        Scalar weighted = 0.0f;
        for (std::size_t keyPos = 0; keyPos <= globalQuery; ++keyPos) {
          const Scalar attnWeight = scores[keyPos] / sumExp;
          const Scalar vValue = cachedValues.At({static_cast<Index>(batchIndex), static_cast<Index>(keyPos),
                                                 static_cast<Index>(kvHead * headDim + dim)});
          weighted += attnWeight * vValue;
        }
        merged.At({static_cast<Index>(localQuery), static_cast<Index>(head * headDim + dim)}) = weighted;
      }
    }
  }

  return output_.Forward(merged, output);
}

Status MultiHeadAttention::ForwardWithBatchPaddedCache(const Tensor& input, const RopeCache& rope,
                                                     inference::BatchKvCache& cache, const std::size_t layerIndex,
                                                     const std::size_t maxSeqLen,
                                                     const std::vector<std::size_t>& seqLens,
                                                     const std::vector<bool>& active, Tensor& output) const {
  if (input.Rank() != 2 || input.GetShape()[1] != config_.hiddenDim) {
    return Status::Fail(ErrorCode::InvalidArgument, "padded batch attention input must be [batch*seq, hidden_dim]");
  }

  const std::size_t batchSize = seqLens.size();
  if (active.size() != batchSize || cache.BatchSize() != batchSize) {
    return Status::Fail(ErrorCode::InvalidArgument, "padded batch attention size mismatch");
  }

  if (input.GetShape()[0] != static_cast<Dimension>(batchSize * maxSeqLen)) {
    return Status::Fail(ErrorCode::InvalidArgument, "padded batch attention flat seq mismatch");
  }

  Tensor queries;
  Tensor keys;
  Tensor values;
  const Status queryStatus = query_.Forward(input, queries);
  const Status keyStatus = key_.Forward(input, keys);
  const Status valueStatus = value_.Forward(input, values);
  if (!queryStatus.IsOk() || !keyStatus.IsOk() || !valueStatus.IsOk()) {
    return Status::Fail(ErrorCode::Internal, "padded batch attention projection failed");
  }

  Tensor merged = Tensor::Zeros(input.GetShape());

  for (std::size_t batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
    if (!active[batchIndex] || seqLens[batchIndex] == 0) {
      continue;
    }

    const Dimension seqLen = static_cast<Dimension>(seqLens[batchIndex]);
    const Dimension flatOffset = static_cast<Dimension>(batchIndex * maxSeqLen);
    const Dimension kvDim = keys.GetShape()[1];

    Tensor slotQueries = Tensor::Zeros(Shape{seqLen, config_.hiddenDim});
    Tensor slotKeys = Tensor::Zeros(Shape{seqLen, kvDim});
    Tensor slotValues = Tensor::Zeros(Shape{seqLen, kvDim});
    for (Dimension row = 0; row < seqLen; ++row) {
      for (Dimension dim = 0; dim < config_.hiddenDim; ++dim) {
        slotQueries.At({static_cast<Index>(row), static_cast<Index>(dim)}) =
            queries.At({static_cast<Index>(flatOffset + row), static_cast<Index>(dim)});
      }
      for (Dimension dim = 0; dim < kvDim; ++dim) {
        slotKeys.At({static_cast<Index>(row), static_cast<Index>(dim)}) =
            keys.At({static_cast<Index>(flatOffset + row), static_cast<Index>(dim)});
        slotValues.At({static_cast<Index>(row), static_cast<Index>(dim)}) =
            values.At({static_cast<Index>(flatOffset + row), static_cast<Index>(dim)});
      }
    }

    ApplyRope(slotQueries, rope, config_.numHeads, config_.hiddenDim / config_.numHeads, 0);
    ApplyRope(slotKeys, rope, config_.numKvHeads, config_.hiddenDim / config_.numHeads, 0);

    const Status writeStatus = cache.Write(layerIndex, batchIndex, 0, slotKeys, slotValues);
    if (!writeStatus.IsOk()) {
      return writeStatus;
    }
    cache.SetLength(batchIndex, seqLens[batchIndex]);

    const Tensor& cachedKeys = cache.Keys(layerIndex, batchIndex);
    const Tensor& cachedValues = cache.Values(layerIndex, batchIndex);
    const Dimension numHeads = config_.numHeads;
    const Dimension numKvHeads = config_.numKvHeads;
    const Dimension headDim = config_.hiddenDim / numHeads;
    const Dimension headsPerKv = numHeads / numKvHeads;
    const Scalar scale = 1.0f / std::sqrt(static_cast<Scalar>(headDim));
    const Scalar maskValue = -std::numeric_limits<Scalar>::infinity();

    Tensor slotMerged = Tensor::Zeros(Shape{seqLen, config_.hiddenDim});
    for (Dimension localQuery = 0; localQuery < seqLen; ++localQuery) {
      const std::size_t globalQuery = static_cast<std::size_t>(localQuery);
      for (Dimension head = 0; head < numHeads; ++head) {
        const Dimension kvHead = head / headsPerKv;
        std::vector<Scalar> scores(globalQuery + 1, maskValue);
        for (std::size_t keyPos = 0; keyPos <= globalQuery; ++keyPos) {
          Scalar dot = 0.0f;
          for (Dimension dim = 0; dim < headDim; ++dim) {
            dot += slotQueries.At({static_cast<Index>(localQuery), static_cast<Index>(head * headDim + dim)}) *
                   cachedKeys.At({static_cast<Index>(batchIndex), static_cast<Index>(keyPos),
                                  static_cast<Index>(kvHead * headDim + dim)});
          }
          scores[keyPos] = dot * scale;
        }

        Scalar maxScore = scores[0];
        for (std::size_t keyPos = 1; keyPos <= globalQuery; ++keyPos) {
          maxScore = std::max(maxScore, scores[keyPos]);
        }

        Scalar sumExp = 0.0f;
        for (std::size_t keyPos = 0; keyPos <= globalQuery; ++keyPos) {
          scores[keyPos] = std::exp(scores[keyPos] - maxScore);
          sumExp += scores[keyPos];
        }

        for (Dimension dim = 0; dim < headDim; ++dim) {
          Scalar weighted = 0.0f;
          for (std::size_t keyPos = 0; keyPos <= globalQuery; ++keyPos) {
            weighted += (scores[keyPos] / sumExp) *
                         cachedValues.At({static_cast<Index>(batchIndex), static_cast<Index>(keyPos),
                                          static_cast<Index>(kvHead * headDim + dim)});
          }
          slotMerged.At({static_cast<Index>(localQuery), static_cast<Index>(head * headDim + dim)}) = weighted;
        }
      }
    }

    for (Dimension row = 0; row < seqLen; ++row) {
      for (Dimension dim = 0; dim < config_.hiddenDim; ++dim) {
        merged.At({static_cast<Index>(flatOffset + row), static_cast<Index>(dim)}) =
            slotMerged.At({static_cast<Index>(row), static_cast<Index>(dim)});
      }
    }
  }

  return output_.Forward(merged, output);
}

} // namespace llm::model
