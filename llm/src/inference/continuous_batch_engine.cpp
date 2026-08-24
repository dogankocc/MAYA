#include "llm/inference/continuous_batch_engine.hpp"

#include "llm/inference/sampling/sampler.hpp"

namespace llm::inference {

ContinuousBatchEngine::ContinuousBatchEngine(const model::TransformerModel& model, const std::size_t maxSlots)
    : maxSlots_(maxSlots), slots_(maxSlots), admittedDecodeTokens_(maxSlots), engine_(model, maxSlots) {}

void ContinuousBatchEngine::Reset() {
  for (BatchSlot& slot : slots_) {
    slot.state = SlotState::Empty;
    slot.tokens.clear();
  }
  pending_.clear();
  admittedDecodeTokens_.assign(maxSlots_, std::nullopt);
  engine_.Reset();
}

std::optional<std::size_t> ContinuousBatchEngine::FindEmptySlot() const {
  for (std::size_t slotId = 0; slotId < slots_.size(); ++slotId) {
    if (slots_[slotId].state == SlotState::Empty) {
      return slotId;
    }
  }
  return std::nullopt;
}

Status ContinuousBatchEngine::PrefillSlot(const std::size_t slotId, const std::vector<TokenId>& prompt,
                                          Tensor& rowLogits) {
  const Status prefillStatus = engine_.PrefillSlot(slotId, prompt, rowLogits);
  if (!prefillStatus.IsOk()) {
    return prefillStatus;
  }

  slots_[slotId].tokens = prompt;
  slots_[slotId].state = SlotState::Decoding;
  return Status::Ok();
}

Status ContinuousBatchEngine::AdmitPending() {
  while (!pending_.empty()) {
    const std::optional<std::size_t> slotId = FindEmptySlot();
    if (!slotId.has_value()) {
      break;
    }

    const std::vector<TokenId> prompt = std::move(pending_.front());
    pending_.pop_front();

    Tensor rowLogits = Tensor::Zeros(Shape{1, engine_.GetModelConfig().vocabSize});
    const Status status = PrefillSlot(slotId.value(), prompt, rowLogits);
    if (!status.IsOk()) {
      return status;
    }

    admittedDecodeTokens_[slotId.value()] = Sampler::SampleGreedy(rowLogits);
  }

  return Status::Ok();
}

Status ContinuousBatchEngine::Enqueue(const std::vector<TokenId>& prompt, std::size_t& slotId, Tensor& prefillLogits) {
  if (prompt.empty()) {
    return Status::Fail(ErrorCode::InvalidArgument, "continuous batch prompt cannot be empty");
  }

  const std::optional<std::size_t> emptySlot = FindEmptySlot();
  if (emptySlot.has_value()) {
    slotId = emptySlot.value();
    return PrefillSlot(slotId, prompt, prefillLogits);
  }

  pending_.push_back(prompt);
  slotId = maxSlots_;
  prefillLogits = Tensor::Zeros(Shape{1, engine_.GetModelConfig().vocabSize});
  return Status::Ok();
}

Status ContinuousBatchEngine::Step(const std::vector<TokenId>& nextTokens, Tensor& logits,
                                   std::vector<bool>& finishedMask) {
  if (nextTokens.size() != maxSlots_) {
    return Status::Fail(ErrorCode::InvalidArgument, "continuous batch token size mismatch");
  }

  const Status admitStatus = AdmitPending();
  if (!admitStatus.IsOk()) {
    return admitStatus;
  }

  finishedMask.assign(maxSlots_, false);

  std::vector<TokenId> effectiveTokens = nextTokens;
  for (std::size_t slotId = 0; slotId < maxSlots_; ++slotId) {
    if (admittedDecodeTokens_[slotId].has_value()) {
      effectiveTokens[slotId] = admittedDecodeTokens_[slotId].value();
      admittedDecodeTokens_[slotId] = std::nullopt;
    }
  }

  std::vector<bool> active(maxSlots_, false);
  bool hasActive = false;

  for (std::size_t slotId = 0; slotId < maxSlots_; ++slotId) {
    if (slots_[slotId].state == SlotState::Decoding) {
      active[slotId] = true;
      hasActive = true;
    }
  }

  if (!hasActive) {
    logits = Tensor::Zeros(Shape{maxSlots_, engine_.GetModelConfig().vocabSize});
    return Status::Ok();
  }

  engine_.SetActive(active);
  logits = Tensor::Zeros(Shape{maxSlots_, engine_.GetModelConfig().vocabSize});
  const Status decodeStatus = engine_.DecodeBatch(effectiveTokens, logits);
  if (!decodeStatus.IsOk()) {
    return decodeStatus;
  }

  for (std::size_t slotId = 0; slotId < maxSlots_; ++slotId) {
    if (!active[slotId]) {
      continue;
    }

    slots_[slotId].tokens.push_back(effectiveTokens[slotId]);

    if (effectiveTokens[slotId] == eosTokenId_ ||
        engine_.SequenceLength(slotId) >= engine_.GetModelConfig().maxSeqLen) {
      slots_[slotId].state = SlotState::Empty;
      slots_[slotId].tokens.clear();
      finishedMask[slotId] = true;
    }
  }

  return Status::Ok();
}

} // namespace llm::inference
