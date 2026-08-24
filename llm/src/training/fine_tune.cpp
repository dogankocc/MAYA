#include "llm/training/fine_tune.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

#include "llm/model/checkpoint/checkpoint.hpp"
#include "llm/model/transformer.hpp"
#include "llm/quantization/quant_checkpoint.hpp"
#include "llm/tokenizer/bpe_tokenizer.hpp"
#include "llm/training/chat_corpus.hpp"
#include "llm/training/trainer.hpp"

namespace llm::training {

namespace {

[[nodiscard]] Result<model::TransformerModel> LoadTrainableModel(const std::string& inputCheckpointPath) {
  if (inputCheckpointPath.ends_with(".ckptq")) {
    return quantization::QuantCheckpoint::Load(inputCheckpointPath);
  }
  return model::Checkpoint::Load(inputCheckpointPath);
}

} // namespace

Status RunFineTuneOnCorpus(const FineTuneOptions& options) {
  const auto samplesResult = LoadDialogueCorpus(options.corpusPath);
  if (!samplesResult.IsOk()) {
    return Status::Fail(ErrorCode::InvalidArgument, "auto-learn corpus could not be loaded");
  }

  const std::vector<DialogueSample> samples = samplesResult.Value();
  if (samples.empty()) {
    return Status::Fail(ErrorCode::InvalidArgument, "auto-learn corpus is empty");
  }

  const auto tokenizerResult = BpeTokenizer::Load(options.tokenizerPath);
  if (!tokenizerResult.IsOk()) {
    return Status::Fail(ErrorCode::InvalidArgument, "tokenizer could not be loaded for fine-tune");
  }
  BpeTokenizer tokenizer = tokenizerResult.Value();

  const auto modelResult = LoadTrainableModel(options.inputCheckpointPath);
  if (!modelResult.IsOk()) {
    return Status::Fail(ErrorCode::InvalidArgument, "checkpoint could not be loaded for fine-tune");
  }

  model::TransformerModel model = modelResult.Value();
  const std::size_t maxSeqLen = model.GetConfig().maxSeqLen;
  std::vector<std::vector<TokenId>> batches = BuildTrainingBatches(tokenizer, samples, maxSeqLen);
  if (batches.empty()) {
    return Status::Fail(ErrorCode::InvalidArgument, "no training batches for fine-tune");
  }

  const std::size_t steps =
      std::min(options.maxSteps, std::max<std::size_t>(80, samples.size() * options.stepsPerSample));

  TrainerConfig trainerConfig;
  trainerConfig.optimizer.learningRate = options.learningRate;
  Trainer trainer(model, trainerConfig);

  std::mt19937 rng(1337);
  float lastLoss = 0.0f;
  std::size_t sampleIndex = 0;
  std::cout << "[auto-learn] fine-tune samples=" << samples.size() << " steps=" << steps << '\n';

  for (std::size_t step = 0; step < steps; ++step) {
    if (step > 0 && step % batches.size() == 0) {
      std::shuffle(batches.begin(), batches.end(), rng);
    }

    const Status status = trainer.TrainStep(batches[sampleIndex % batches.size()], lastLoss);
    sampleIndex += 1;
    if (!status.IsOk()) {
      return status;
    }

    if ((step + 1) % 100 == 0 || step + 1 == steps) {
      std::cout << "[auto-learn] step " << (step + 1) << '/' << steps << " loss=" << lastLoss << '\n';
    }
  }

  if (!model::Checkpoint::Save(model, options.fp32OutputPath).IsOk()) {
    return Status::Fail(ErrorCode::Internal, "failed to save fine-tuned checkpoint");
  }

  if (!quantization::QuantCheckpoint::ConvertFile(options.fp32OutputPath, options.quantOutputPath).IsOk()) {
    return Status::Fail(ErrorCode::Internal, "failed to quantize fine-tuned checkpoint");
  }

  std::cout << "[auto-learn] updated " << options.quantOutputPath << '\n';
  return Status::Ok();
}

} // namespace llm::training
