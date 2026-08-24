#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/tokenizer/vocabulary.hpp"
#include "llm/inference/batch_inference_engine.hpp"
#include "llm/model/transformer.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::inference {

enum class SlotState { Empty, Decoding, Finished };

struct BatchSlot {
  SlotState state = SlotState::Empty;
  std::vector<TokenId> tokens;
};

class ContinuousBatchEngine {
public:
  ContinuousBatchEngine(const model::TransformerModel& model, std::size_t maxSlots);

  [[nodiscard]] Status Enqueue(const std::vector<TokenId>& prompt, std::size_t& slotId, Tensor& prefillLogits);

  [[nodiscard]] Status Step(const std::vector<TokenId>& nextTokens, Tensor& logits, std::vector<bool>& finishedMask);

  [[nodiscard]] std::size_t MaxSlots() const { return maxSlots_; }

  [[nodiscard]] std::size_t PendingCount() const { return pending_.size(); }

  [[nodiscard]] const std::vector<BatchSlot>& Slots() const { return slots_; }

  [[nodiscard]] BatchInferenceEngine& Engine() { return engine_; }

  void Reset();

private:
  [[nodiscard]] Status AdmitPending();

  [[nodiscard]] std::optional<std::size_t> FindEmptySlot() const;

  [[nodiscard]] Status PrefillSlot(std::size_t slotId, const std::vector<TokenId>& prompt, Tensor& rowLogits);

  std::size_t maxSlots_ = 0;
  std::vector<BatchSlot> slots_;
  std::deque<std::vector<TokenId>> pending_;
  std::vector<std::optional<TokenId>> admittedDecodeTokens_;
  BatchInferenceEngine engine_;
  TokenId eosTokenId_ = kEosTokenId;
};

} // namespace llm::inference
