#pragma once

#include <random>

#include "llm/core/config.hpp"
#include "llm/core/status.hpp"
#include "llm/model/linear.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::model {

class SwiGluFfn {
public:
  explicit SwiGluFfn(const ModelConfig& config);

  void ResetParameters(std::mt19937& rng);

  [[nodiscard]] Status Forward(const Tensor& input, Tensor& output) const;

  [[nodiscard]] Linear& GateProjection() { return gate_; }

  [[nodiscard]] Linear& UpProjection() { return up_; }

  [[nodiscard]] Linear& DownProjection() { return down_; }

  [[nodiscard]] const Linear& GateProjection() const { return gate_; }

  [[nodiscard]] const Linear& UpProjection() const { return up_; }

  [[nodiscard]] const Linear& DownProjection() const { return down_; }

private:
  ModelConfig config_;
  Linear gate_;
  Linear up_;
  Linear down_;
};

} // namespace llm::model
