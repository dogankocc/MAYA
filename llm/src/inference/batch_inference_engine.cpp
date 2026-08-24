#include "llm/inference/batch_inference_engine.hpp"

#include <algorithm>

#include "llm/model/rms_norm.hpp"
#include "llm/tensor/ops.hpp"

namespace llm::inference {

BatchInferenceEngine::BatchInferenceEngine(const model::TransformerModel& model, const std::size_t batchSize)
    : model_(model), batchSize_(batchSize), active_(batchSize, true), cache_(model.GetConfig(), batchSize) {}

void BatchInferenceEngine::Reset() {
  cache_.Reset();
  std::fill(active_.begin(), active_.end(), true);
}

void BatchInferenceEngine::SetActive(const std::vector<bool>& active) {
  if (active.size() != batchSize_) {
    return;
  }
  active_ = active;
}

std::size_t BatchInferenceEngine::SequenceLength(const std::size_t batchIndex) const {
  return cache_.Length(batchIndex);
}

Status BatchInferenceEngine::EmbedTokens(const std::vector<TokenId>& tokens, Tensor& hidden) const {
  if (tokens.empty()) {
    return Status::Fail(ErrorCode::InvalidArgument, "embed requires at least one token");
  }

  const auto& config = model_.GetConfig();
  const Dimension seqLen = tokens.size();
  hidden = Tensor::Zeros(Shape{seqLen, config.hiddenDim});

  for (Dimension position = 0; position < seqLen; ++position) {
    const TokenId token = tokens[static_cast<std::size_t>(position)];
    if (token >= config.vocabSize) {
      return Status::Fail(ErrorCode::OutOfRange, "token id exceeds vocab_size");
    }

    for (Dimension dim = 0; dim < config.hiddenDim; ++dim) {
      hidden.At({static_cast<Index>(position), static_cast<Index>(dim)}) =
          model_.TokenEmbedding().At({static_cast<Index>(token), static_cast<Index>(dim)});
    }
  }

  return Status::Ok();
}

Status BatchInferenceEngine::EmbedBatch(const std::vector<TokenId>& tokens, Tensor& hidden) const {
  if (tokens.size() != batchSize_) {
    return Status::Fail(ErrorCode::InvalidArgument, "decode batch token count mismatch");
  }

  const auto& config = model_.GetConfig();
  hidden = Tensor::Zeros(Shape{batchSize_, config.hiddenDim});

  for (std::size_t batchIndex = 0; batchIndex < batchSize_; ++batchIndex) {
    if (!active_[batchIndex]) {
      continue;
    }

    const TokenId token = tokens[batchIndex];
    if (token >= config.vocabSize) {
      return Status::Fail(ErrorCode::OutOfRange, "token id exceeds vocab_size");
    }

    for (Dimension dim = 0; dim < config.hiddenDim; ++dim) {
      hidden.At({static_cast<Index>(batchIndex), static_cast<Index>(dim)}) =
          model_.TokenEmbedding().At({static_cast<Index>(token), static_cast<Index>(dim)});
    }
  }

  return Status::Ok();
}

Status BatchInferenceEngine::ForwardHiddenSlot(const Tensor& hidden, const std::size_t batchIndex,
                                               const std::size_t cacheStart, Tensor& output) {
  const Dimension seqLen = hidden.GetShape()[0];
  Tensor current = hidden;

  for (std::size_t layer = 0; layer < model_.NumLayers(); ++layer) {
    Tensor next = Tensor::Zeros(Shape{seqLen, model_.GetConfig().hiddenDim});
    const Status layerStatus = model_.Layer(layer).ForwardWithBatchSlotCache(
        current, model_.GetRopeCache(), cache_, layer, batchIndex, cacheStart, next);
    if (!layerStatus.IsOk()) {
      return layerStatus;
    }
    current = std::move(next);
  }

  output = std::move(current);
  return Status::Ok();
}

Status BatchInferenceEngine::ForwardHiddenBatch(const Tensor& hidden, const std::vector<std::size_t>& cacheStarts,
                                                Tensor& output) {
  if (hidden.GetShape()[0] != static_cast<Dimension>(batchSize_)) {
    return Status::Fail(ErrorCode::InvalidArgument, "batch hidden row count mismatch");
  }

  Tensor current = hidden;

  for (std::size_t layer = 0; layer < model_.NumLayers(); ++layer) {
    Tensor next = Tensor::Zeros(Shape{batchSize_, model_.GetConfig().hiddenDim});
    const Status layerStatus = model_.Layer(layer).ForwardWithBatchCache(current, model_.GetRopeCache(), cache_, layer,
                                                                         cacheStarts, active_, next);
    if (!layerStatus.IsOk()) {
      return layerStatus;
    }
    current = std::move(next);
  }

  output = std::move(current);
  return Status::Ok();
}

Status BatchInferenceEngine::LogitsFromRows(const Tensor& hiddenRows, Tensor& logits) const {
  if (hiddenRows.Rank() != 2 || hiddenRows.GetShape()[0] != static_cast<Dimension>(batchSize_)) {
    return Status::Fail(ErrorCode::InvalidArgument, "batch logits expect hidden rows [batch, hidden_dim]");
  }

  Tensor normalized;
  const Status normStatus =
      model::RmsNorm(hiddenRows, model_.FinalNormWeight(), model_.GetConfig().normEps, normalized);
  if (!normStatus.IsOk()) {
    return normStatus;
  }

  return model_.LmHeadForward(normalized, logits);
}

Status BatchInferenceEngine::LogitsFromSlotLastRow(const Tensor& hidden, Tensor& logitsRow) const {
  if (hidden.Rank() != 2 || hidden.GetShape()[1] != model_.GetConfig().hiddenDim) {
    return Status::Fail(ErrorCode::InvalidArgument, "slot logits expect hidden [seq, hidden_dim]");
  }

  const Dimension lastIndex = hidden.GetShape()[0] - 1;
  Tensor lastRow = Tensor::Zeros(Shape{1, model_.GetConfig().hiddenDim});
  for (Dimension dim = 0; dim < model_.GetConfig().hiddenDim; ++dim) {
    lastRow.At({0, static_cast<Index>(dim)}) =
        hidden.At({static_cast<Index>(lastIndex), static_cast<Index>(dim)});
  }

  Tensor logits = Tensor::Zeros(Shape{1, model_.GetConfig().vocabSize});
  Tensor normalized;
  const Status normStatus =
      model::RmsNorm(lastRow, model_.FinalNormWeight(), model_.GetConfig().normEps, normalized);
  if (!normStatus.IsOk()) {
    return normStatus;
  }

  const Status matmulStatus = model_.LmHeadForward(normalized, logits);
  if (!matmulStatus.IsOk()) {
    return matmulStatus;
  }

  logitsRow = std::move(logits);
  return Status::Ok();
}

Status BatchInferenceEngine::EmbedPaddedBatch(const std::vector<std::vector<TokenId>>& prompts,
                                              const std::size_t maxSeqLen, Tensor& hidden,
                                              std::vector<std::size_t>& seqLens) const {
  if (prompts.size() != batchSize_) {
    return Status::Fail(ErrorCode::InvalidArgument, "padded embed batch size mismatch");
  }

  const auto& config = model_.GetConfig();
  hidden = Tensor::Zeros(Shape{batchSize_ * maxSeqLen, config.hiddenDim});
  seqLens.assign(batchSize_, 0);

  for (std::size_t batchIndex = 0; batchIndex < batchSize_; ++batchIndex) {
    if (!active_[batchIndex]) {
      continue;
    }

    const std::vector<TokenId>& prompt = prompts[batchIndex];
    seqLens[batchIndex] = prompt.size();

    for (std::size_t position = 0; position < prompt.size(); ++position) {
      const TokenId token = prompt[position];
      const Dimension flatRow = static_cast<Dimension>(batchIndex * maxSeqLen + position);
      for (Dimension dim = 0; dim < config.hiddenDim; ++dim) {
        hidden.At({static_cast<Index>(flatRow), static_cast<Index>(dim)}) =
            model_.TokenEmbedding().At({static_cast<Index>(token), static_cast<Index>(dim)});
      }
    }
  }

  return Status::Ok();
}

Status BatchInferenceEngine::ForwardHiddenPaddedBatch(const Tensor& hidden, const std::size_t maxSeqLen,
                                                    const std::vector<std::size_t>& seqLens, Tensor& output) {
  Tensor current = hidden;

  for (std::size_t layer = 0; layer < model_.NumLayers(); ++layer) {
    Tensor next = Tensor::Zeros(hidden.GetShape());
    const Status layerStatus = model_.Layer(layer).ForwardWithBatchPaddedCache(
        current, model_.GetRopeCache(), cache_, layer, maxSeqLen, seqLens, active_, next);
    if (!layerStatus.IsOk()) {
      return layerStatus;
    }
    current = std::move(next);
  }

  output = std::move(current);
  return Status::Ok();
}

Status BatchInferenceEngine::LogitsFromPaddedRows(const Tensor& hidden, const std::size_t maxSeqLen,
                                                const std::vector<std::size_t>& seqLens, Tensor& logits) const {
  const Dimension vocabSize = model_.GetConfig().vocabSize;
  logits = Tensor::Zeros(Shape{batchSize_, vocabSize});

  for (std::size_t batchIndex = 0; batchIndex < batchSize_; ++batchIndex) {
    if (!active_[batchIndex] || seqLens[batchIndex] == 0) {
      continue;
    }

    const Dimension flatRow = static_cast<Dimension>(batchIndex * maxSeqLen + seqLens[batchIndex] - 1);
    Tensor row = Tensor::Zeros(Shape{1, model_.GetConfig().hiddenDim});
    for (Dimension dim = 0; dim < model_.GetConfig().hiddenDim; ++dim) {
      row.At({0, static_cast<Index>(dim)}) = hidden.At({static_cast<Index>(flatRow), static_cast<Index>(dim)});
    }

    Tensor rowLogits = Tensor::Zeros(Shape{1, vocabSize});
    const Status status = LogitsFromSlotLastRow(row, rowLogits);
    if (!status.IsOk()) {
      return status;
    }

    for (Dimension vocab = 0; vocab < vocabSize; ++vocab) {
      logits.At({static_cast<Index>(batchIndex), static_cast<Index>(vocab)}) = rowLogits.At({0, static_cast<Index>(vocab)});
    }
  }

  return Status::Ok();
}

Status BatchInferenceEngine::PrefillBatchPadded(const std::vector<std::vector<TokenId>>& prompts, Tensor& logits) {
  if (prompts.size() != batchSize_) {
    return Status::Fail(ErrorCode::InvalidArgument, "prefill batch size mismatch");
  }

  Reset();

  std::size_t maxSeqLen = 0;
  for (const std::vector<TokenId>& prompt : prompts) {
    if (!prompt.empty()) {
      maxSeqLen = std::max(maxSeqLen, prompt.size());
    }
  }

  if (maxSeqLen == 0) {
    return Status::Fail(ErrorCode::InvalidArgument, "prefill prompt cannot be empty");
  }

  if (maxSeqLen > model_.GetConfig().maxSeqLen) {
    return Status::Fail(ErrorCode::InvalidArgument, "prefill exceeds max_seq_len");
  }

  Tensor hidden;
  std::vector<std::size_t> seqLens;
  const Status embedStatus = EmbedPaddedBatch(prompts, maxSeqLen, hidden, seqLens);
  if (!embedStatus.IsOk()) {
    return embedStatus;
  }

  Tensor output;
  const Status forwardStatus = ForwardHiddenPaddedBatch(hidden, maxSeqLen, seqLens, output);
  if (!forwardStatus.IsOk()) {
    return forwardStatus;
  }

  return LogitsFromPaddedRows(output, maxSeqLen, seqLens, logits);
}

Status BatchInferenceEngine::PrefillSlot(const std::size_t batchIndex, const std::vector<TokenId>& prompt,
                                         Tensor& rowLogits) {
  if (batchIndex >= batchSize_) {
    return Status::Fail(ErrorCode::OutOfRange, "prefill slot index out of range");
  }

  if (prompt.empty()) {
    return Status::Fail(ErrorCode::InvalidArgument, "prefill prompt cannot be empty");
  }

  if (prompt.size() > model_.GetConfig().maxSeqLen) {
    return Status::Fail(ErrorCode::InvalidArgument, "prefill exceeds max_seq_len");
  }

  cache_.SetLength(batchIndex, 0);

  Tensor hidden;
  const Status embedStatus = EmbedTokens(prompt, hidden);
  if (!embedStatus.IsOk()) {
    return embedStatus;
  }

  Tensor output;
  const Status forwardStatus = ForwardHiddenSlot(hidden, batchIndex, 0, output);
  if (!forwardStatus.IsOk()) {
    return forwardStatus;
  }

  return LogitsFromSlotLastRow(output, rowLogits);
}

Status BatchInferenceEngine::PrefillBatch(const std::vector<std::vector<TokenId>>& prompts, Tensor& logits) {
  if (usePaddedPrefill_) {
    return PrefillBatchPadded(prompts, logits);
  }

  if (prompts.size() != batchSize_) {
    return Status::Fail(ErrorCode::InvalidArgument, "prefill batch size mismatch");
  }

  Reset();

  const Dimension vocabSize = model_.GetConfig().vocabSize;
  logits = Tensor::Zeros(Shape{batchSize_, vocabSize});

  for (std::size_t batchIndex = 0; batchIndex < batchSize_; ++batchIndex) {
    if (!active_[batchIndex]) {
      continue;
    }

    const std::vector<TokenId>& prompt = prompts[batchIndex];
    if (prompt.empty()) {
      return Status::Fail(ErrorCode::InvalidArgument, "prefill prompt cannot be empty");
    }

    if (prompt.size() > model_.GetConfig().maxSeqLen) {
      return Status::Fail(ErrorCode::InvalidArgument, "prefill exceeds max_seq_len");
    }

    Tensor hidden;
    const Status embedStatus = EmbedTokens(prompt, hidden);
    if (!embedStatus.IsOk()) {
      return embedStatus;
    }

    Tensor output;
    const Status forwardStatus = ForwardHiddenSlot(hidden, batchIndex, 0, output);
    if (!forwardStatus.IsOk()) {
      return forwardStatus;
    }

    Tensor rowLogits = Tensor::Zeros(Shape{1, vocabSize});
    const Status logitsStatus = LogitsFromSlotLastRow(output, rowLogits);
    if (!logitsStatus.IsOk()) {
      return logitsStatus;
    }

    for (Dimension vocab = 0; vocab < vocabSize; ++vocab) {
      logits.At({static_cast<Index>(batchIndex), static_cast<Index>(vocab)}) = rowLogits.At({0, static_cast<Index>(vocab)});
    }
  }

  return Status::Ok();
}

Status BatchInferenceEngine::DecodeBatch(const std::vector<TokenId>& tokens, Tensor& logits) {
  if (tokens.size() != batchSize_) {
    return Status::Fail(ErrorCode::InvalidArgument, "decode batch token count mismatch");
  }

  const Dimension vocabSize = model_.GetConfig().vocabSize;
  logits = Tensor::Zeros(Shape{batchSize_, vocabSize});

  std::vector<std::size_t> cacheStarts(batchSize_, 0);
  bool hasActive = false;

  for (std::size_t batchIndex = 0; batchIndex < batchSize_; ++batchIndex) {
    if (!active_[batchIndex]) {
      continue;
    }

    if (cache_.Length(batchIndex) >= model_.GetConfig().maxSeqLen) {
      return Status::Fail(ErrorCode::OutOfRange, "decode exceeds max_seq_len");
    }

    cacheStarts[batchIndex] = cache_.Length(batchIndex);
    hasActive = true;
  }

  if (!hasActive) {
    return Status::Ok();
  }

  Tensor hidden;
  const Status embedStatus = EmbedBatch(tokens, hidden);
  if (!embedStatus.IsOk()) {
    return embedStatus;
  }

  Tensor output;
  const Status forwardStatus = ForwardHiddenBatch(hidden, cacheStarts, output);
  if (!forwardStatus.IsOk()) {
    return forwardStatus;
  }

  Tensor batchLogits = Tensor::Zeros(Shape{batchSize_, vocabSize});
  const Status logitsStatus = LogitsFromRows(output, batchLogits);
  if (!logitsStatus.IsOk()) {
    return logitsStatus;
  }

  for (std::size_t batchIndex = 0; batchIndex < batchSize_; ++batchIndex) {
    if (!active_[batchIndex]) {
      continue;
    }

    for (Dimension vocab = 0; vocab < vocabSize; ++vocab) {
      logits.At({static_cast<Index>(batchIndex), static_cast<Index>(vocab)}) =
          batchLogits.At({static_cast<Index>(batchIndex), static_cast<Index>(vocab)});
    }
  }

  return Status::Ok();
}

} // namespace llm::inference
