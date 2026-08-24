#include "llm/training/adamw.hpp"

#include <cmath>

namespace llm::training {

AdamW::AdamW(AdamWConfig config) : config_(config) {}

void AdamW::Reset() {
  step_ = 0;
  moment1_.clear();
  moment2_.clear();
}

Status AdamW::Step(ParameterList& parameters) {
  if (moment1_.size() != parameters.Size()) {
    moment1_.clear();
    moment2_.clear();
    for (const Parameter& parameter : parameters.Parameters()) {
      moment1_.push_back(Tensor::Zeros(parameter.tensor->GetShape()));
      moment2_.push_back(Tensor::Zeros(parameter.tensor->GetShape()));
    }
  }

  ++step_;
  const float biasCorrection1 = 1.0f - std::pow(config_.beta1, static_cast<float>(step_));
  const float biasCorrection2 = 1.0f - std::pow(config_.beta2, static_cast<float>(step_));
  const float stepSize = config_.learningRate * std::sqrt(biasCorrection2) / biasCorrection1;

  for (std::size_t index = 0; index < parameters.Size(); ++index) {
    Parameter& parameter = parameters.Parameters()[index];
    Tensor& moment1 = moment1_[index];
    Tensor& moment2 = moment2_[index];

    for (Index element = 0; element < static_cast<Index>(parameter.tensor->Numel()); ++element) {
      const Scalar gradient = parameter.grad[element];
      moment1[element] = config_.beta1 * moment1[element] + (1.0f - config_.beta1) * gradient;
      moment2[element] = config_.beta2 * moment2[element] + (1.0f - config_.beta2) * gradient * gradient;

      const Scalar update = stepSize * moment1[element] / (std::sqrt(moment2[element]) + config_.epsilon);
      (*parameter.tensor)[element] -= update;
      (*parameter.tensor)[element] -= config_.learningRate * config_.weightDecay * (*parameter.tensor)[element];
    }
  }

  return Status::Ok();
}

} // namespace llm::training
