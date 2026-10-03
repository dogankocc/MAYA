#include "llm/training/trainer.hpp"

#include "llm/training/transformer_train.hpp"

namespace llm::training {

Trainer::Trainer(model::TransformerModel& model, TrainerConfig config)
    : model_(model), config_(std::move(config)), optimizer_(config_.optimizer) {
  ParameterList::CollectFromModel(model_, parameters_);
  optimizer_.Reset();
}

Status Trainer::TrainStep(const std::vector<TokenId>& tokens, float& loss) {
  // Backward compatible: single sample with immediate step
  const Status trainStatus = RunTrainBackward(model_, parameters_, tokens, loss, true);
  if (!trainStatus.IsOk()) {
    return trainStatus;
  }
  return optimizer_.Step(parameters_);
}

Status Trainer::TrainStep(const TrainingExample& example, float& loss) {
  const Status status = RunTrainBackward(model_, parameters_, example.tokens, loss, true, example.firstTargetToken);
  if (!status.IsOk()) return status;
  return optimizer_.Step(parameters_);
}

Status Trainer::AccumulateGradients(const std::vector<TokenId>& tokens, float& loss, bool zeroGrad) {
  return RunTrainBackward(model_, parameters_, tokens, loss, zeroGrad);
}

Status Trainer::AccumulateGradients(const TrainingExample& example, float& loss, bool zeroGrad) {
  return RunTrainBackward(model_, parameters_, example.tokens, loss, zeroGrad, example.firstTargetToken);
}

Status Trainer::ApplyStep() {
  return optimizer_.Step(parameters_);
}

void Trainer::ZeroGrad() {
  parameters_.ZeroGrad();
}

Status Trainer::TrainEpoch(const std::vector<std::vector<TokenId>>& batches, float& averageLoss) {
  if (batches.empty()) {
    averageLoss = 0.0f;
    return Status::Ok();
  }

  float totalLoss = 0.0f;
  for (const std::vector<TokenId>& batch : batches) {
    float stepLoss = 0.0f;
    const Status status = TrainStep(batch, stepLoss);
    if (!status.IsOk()) {
      return status;
    }
    totalLoss += stepLoss;
  }

  averageLoss = totalLoss / static_cast<float>(batches.size());
  return Status::Ok();
}

} // namespace llm::training
