#pragma once

#include <vector>

#include "llm/core/status.hpp"
#include "llm/core/types.hpp"
#include "llm/inference/kv_cache.hpp"
#include "llm/model/transformer.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::inference {

class InferenceEngine {
public:
  explicit InferenceEngine(const model::TransformerModel& model);

  void Reset();

  [[nodiscard]] Status Prefill(const std::vector<TokenId>& tokens, Tensor& logits);

  [[nodiscard]] Status Decode(TokenId token, Tensor& logits);

  [[nodiscard]] std::size_t SequenceLength() const { return cache_.Length(); }

  [[nodiscard]] const KvCache& GetCache() const { return cache_; }

  [[nodiscard]] const ModelConfig& GetModelConfig() const { return model_.GetConfig(); }

private:
  [[nodiscard]] Status EmbedTokens(const std::vector<TokenId>& tokens, Tensor& hidden) const;

  [[nodiscard]] Status EmbedToken(TokenId token, Tensor& hidden) const;

  [[nodiscard]] Status ForwardHidden(const Tensor& hidden, std::size_t cacheStart, Tensor& output);

  [[nodiscard]] Status LogitsFromRow(const Tensor& hiddenRow, Tensor& logits) const;

  const model::TransformerModel& model_;
  KvCache cache_;
};

} // namespace llm::inference
