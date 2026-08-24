#include "llm/inference/inference_engine.hpp"

#include "llm/model/rms_norm.hpp"
#include "llm/tensor/ops.hpp"

namespace llm::inference {

InferenceEngine::InferenceEngine(const model::TransformerModel& model) : model_(model), cache_(model.GetConfig()) {}

void InferenceEngine::Reset() {
  cache_.Reset();
}

Status InferenceEngine::EmbedTokens(const std::vector<TokenId>& tokens, Tensor& hidden) const {
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

Status InferenceEngine::EmbedToken(const TokenId token, Tensor& hidden) const {
  return EmbedTokens(std::vector<TokenId>{token}, hidden);
}

Status InferenceEngine::ForwardHidden(const Tensor& hidden, const std::size_t cacheStart, Tensor& output) {
  const Dimension seqLen = hidden.GetShape()[0];
  Tensor current = hidden;

  for (std::size_t layer = 0; layer < model_.NumLayers(); ++layer) {
    Tensor next = Tensor::Zeros(Shape{seqLen, model_.GetConfig().hiddenDim});
    const Status layerStatus =
        model_.Layer(layer).ForwardWithCache(current, model_.GetRopeCache(), cache_, layer, cacheStart, next);
    if (!layerStatus.IsOk()) {
      return layerStatus;
    }
    current = std::move(next);
  }

  output = std::move(current);
  return Status::Ok();
}

Status InferenceEngine::LogitsFromRow(const Tensor& hiddenRow, Tensor& logits) const {
  if (hiddenRow.Rank() != 2 || hiddenRow.GetShape()[0] != 1) {
    return Status::Fail(ErrorCode::InvalidArgument, "logits expect hidden row [1, hidden_dim]");
  }

  Tensor normalized;
  const Status normStatus =
      model::RmsNorm(hiddenRow, model_.FinalNormWeight(), model_.GetConfig().normEps, normalized);
  if (!normStatus.IsOk()) {
    return normStatus;
  }

  return model_.LmHeadForward(normalized, logits);
}

Status InferenceEngine::Prefill(const std::vector<TokenId>& tokens, Tensor& logits) {
  if (tokens.empty()) {
    return Status::Fail(ErrorCode::InvalidArgument, "prefill requires at least one token");
  }

  if (tokens.size() > model_.GetConfig().maxSeqLen) {
    return Status::Fail(ErrorCode::InvalidArgument, "prefill exceeds max_seq_len");
  }

  Reset();

  Tensor hidden;
  const Status embedStatus = EmbedTokens(tokens, hidden);
  if (!embedStatus.IsOk()) {
    return embedStatus;
  }

  Tensor output;
  const Status forwardStatus = ForwardHidden(hidden, 0, output);
  if (!forwardStatus.IsOk()) {
    return forwardStatus;
  }

  Tensor lastRow = Tensor::Zeros(Shape{1, model_.GetConfig().hiddenDim});
  const Dimension lastIndex = static_cast<Dimension>(tokens.size() - 1);
  for (Dimension dim = 0; dim < model_.GetConfig().hiddenDim; ++dim) {
    lastRow.At({0, static_cast<Index>(dim)}) = output.At({static_cast<Index>(lastIndex), static_cast<Index>(dim)});
  }

  return LogitsFromRow(lastRow, logits);
}

Status InferenceEngine::Decode(const TokenId token, Tensor& logits) {
  if (cache_.Length() >= model_.GetConfig().maxSeqLen) {
    return Status::Fail(ErrorCode::OutOfRange, "decode exceeds max_seq_len");
  }

  Tensor hidden;
  const Status embedStatus = EmbedToken(token, hidden);
  if (!embedStatus.IsOk()) {
    return embedStatus;
  }

  const std::size_t cacheStart = cache_.Length();
  Tensor output;
  const Status forwardStatus = ForwardHidden(hidden, cacheStart, output);
  if (!forwardStatus.IsOk()) {
    return forwardStatus;
  }

  return LogitsFromRow(output, logits);
}

} // namespace llm::inference
