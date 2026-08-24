#include "llm/training/trainer.hpp"

#include "llm/training/transformer_train.hpp"

namespace llm::training {

Trainer::Trainer(model::TransformerModel& model, TrainerConfig config)
    : model_(model), config_(std::move(config)), optimizer_(config_.optimizer) {
  ParameterList::CollectFromModel(model_, parameters_);
  optimizer_.Reset();
}

Status Trainer::TrainStep(const std::vector<TokenId>& tokens, float& loss) {
  const Status trainStatus = RunTrainBackward(model_, parameters_, tokens, loss);
  if (!trainStatus.IsOk()) {
    return trainStatus;
  }

  return optimizer_.Step(parameters_);
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
