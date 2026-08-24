#pragma once

#include <random>
#include <vector>

#include "llm/core/config.hpp"
#include "llm/core/status.hpp"
#include "llm/core/types.hpp"
#include "llm/model/rope.hpp"
#include "llm/model/transformer_block.hpp"
#include "llm/quantization/quantize.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::model {

class TransformerModel {
public:
  explicit TransformerModel(const ModelConfig& config);

  void ResetParameters(std::mt19937& rng);

  [[nodiscard]] Status Forward(const std::vector<TokenId>& tokens, Tensor& logits) const;

  [[nodiscard]] const ModelConfig& GetConfig() const { return config_; }

  [[nodiscard]] const RopeCache& GetRopeCache() const { return rope_; }

  [[nodiscard]] Tensor& TokenEmbeddingMutable() { return tokenEmbedding_; }

  [[nodiscard]] Tensor& FinalNormWeightMutable() { return finalNormWeight_; }

  [[nodiscard]] Tensor& LmHeadWeightMutable() { return lmHeadWeight_; }

  [[nodiscard]] Status LoadQuantizedLmHeadWeight(const quantization::QuantizedTensor& source);

  [[nodiscard]] TransformerBlock& Layer(const std::size_t index) { return layers_.at(index); }

  [[nodiscard]] const TransformerBlock& Layer(const std::size_t index) const { return layers_.at(index); }

  [[nodiscard]] const Tensor& TokenEmbedding() const { return tokenEmbedding_; }

  [[nodiscard]] const Tensor& FinalNormWeight() const { return finalNormWeight_; }

  [[nodiscard]] const Tensor& LmHeadWeight() const { return lmHeadWeight_; }

  [[nodiscard]] bool UsesQuantizedLmHead() const { return useQuantizedLmHead_; }

  [[nodiscard]] const quantization::QuantizedTensor& QuantizedLmHeadWeight() const { return quantizedLmHeadWeight_; }

  [[nodiscard]] Status LmHeadForward(const Tensor& normalized, Tensor& logits) const;

  void SetQuantizedRuntimeEnabled(const bool enabled) { quantizedRuntimeEnabled_ = enabled; }

  [[nodiscard]] bool IsQuantizedRuntimeEnabled() const { return quantizedRuntimeEnabled_; }

  [[nodiscard]] std::size_t NumLayers() const { return layers_.size(); }

private:
  ModelConfig config_;
  Tensor tokenEmbedding_;
  Tensor finalNormWeight_;
  Tensor lmHeadWeight_;
  quantization::QuantizedTensor quantizedLmHeadWeight_;
  bool useQuantizedLmHead_ = false;
  bool quantizedRuntimeEnabled_ = false;
  RopeCache rope_;
  std::vector<TransformerBlock> layers_;
};

} // namespace llm::model
