#pragma once

#include <cstddef>
#include <string>

#include "llm/core/status.hpp"
#include "llm/training/parameter.hpp"

namespace llm::training {

struct AdamWConfig {
  float learningRate = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-8f;
  float weightDecay = 0.01f;
};

class AdamW {
public:
  explicit AdamW(AdamWConfig config);

  void Reset();

  [[nodiscard]] Status Step(ParameterList& parameters);

  [[nodiscard]] Status SaveState(const std::string& path) const;

  [[nodiscard]] Status LoadState(const std::string& path, const ParameterList& parameters);

  [[nodiscard]] std::size_t StepCount() const { return step_; }

  void SetLearningRate(float learningRate) { config_.learningRate = learningRate; }

  [[nodiscard]] float LearningRate() const { return config_.learningRate; }

private:
  void EnsureMoments(const ParameterList& parameters);

  [[nodiscard]] bool MomentsMatch(const ParameterList& parameters) const;

  AdamWConfig config_;
  std::size_t step_ = 0;
  std::vector<Tensor> moment1_;
  std::vector<Tensor> moment2_;
};

} // namespace llm::training
