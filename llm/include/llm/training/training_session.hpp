#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "llm/core/config.hpp"
#include "llm/core/status.hpp"

namespace llm::training {

class Trainer;

// Full description of a training job. Everything that influences the produced
// weights is part of the session key; changing any of it starts a fresh session.
struct TrainingJobConfig {
  std::vector<std::string> datasetPaths;
  bool includeBuiltin = true;
  ModelConfig model;
  std::size_t steps = 0;             // 0 = auto (derived from sample count)
  float learningRate = 0.0f;         // 0 = auto
  std::size_t checkpointInterval = 1; // save temp checkpoint every N steps
  std::size_t batchSize = 8;         // mini-batch size (samples per gradient update)
  std::uint32_t seed = 42;
  std::string workDir = "training_work";
  std::string outputModelPath = "model.ckpt";
  std::string outputQuantPath = "model.ckptq";
  std::string outputTokenizerPath = "tokenizer_data";
};

struct TrainingSessionState {
  std::string sessionKey;
  std::size_t completedSteps = 0;
  std::size_t totalSteps = 0;
  float lastLoss = 0.0f;
  std::size_t sampleCount = 0;
  TrainingJobConfig config;
};

struct TrainingProgress {
  std::size_t step = 0;
  std::size_t totalSteps = 0;
  float loss = 0.0f;
  bool checkpointSaved = false;
};

[[nodiscard]] TrainingJobConfig DefaultTrainingJobConfig();

[[nodiscard]] std::string ComputeSessionKey(const TrainingJobConfig& config);

// Flat JSON (single object). `ParseTrainingJobConfig` fills missing keys from `defaults`.
[[nodiscard]] std::string SerializeTrainingJobConfigFields(const TrainingJobConfig& config);
[[nodiscard]] TrainingJobConfig ParseTrainingJobConfig(const std::string& json, const TrainingJobConfig& defaults);

[[nodiscard]] std::string SerializeSessionState(const TrainingSessionState& state);
[[nodiscard]] std::optional<TrainingSessionState> ParseSessionState(const std::string& json);

[[nodiscard]] std::string SessionStatePath(const std::string& workDir);
[[nodiscard]] std::string SessionCheckpointPath(const std::string& workDir);
[[nodiscard]] std::string SessionTokenizerPath(const std::string& workDir);
[[nodiscard]] std::string SessionOptimizerPath(const std::string& workDir);

[[nodiscard]] std::optional<TrainingSessionState> LoadSessionState(const std::string& workDir);
[[nodiscard]] Status SaveSessionState(const TrainingSessionState& state);
void ClearWorkDir(const std::string& workDir);

[[nodiscard]] std::size_t ResolveTrainingSteps(std::size_t requestedSteps, std::size_t sampleCount);
[[nodiscard]] float ResolveLearningRate(float requestedRate, std::size_t sampleCount);

// Runs a job end-to-end with periodic temp checkpoints. If a compatible session
// exists in `workDir` it resumes from the last saved step; otherwise the work dir is
// cleared and training starts from scratch. On completion the final model, quantized
// model and tokenizer are published to the configured output paths.
class TrainingRunner {
public:
  using LogFn = std::function<void(const std::string&)>;
  using ProgressFn = std::function<void(const TrainingProgress&)>;

  TrainingRunner(TrainingJobConfig config, LogFn log, ProgressFn progress = nullptr);

  [[nodiscard]] Status Run(const std::atomic<bool>& stopRequested);

  [[nodiscard]] std::size_t ResumedFromStep() const { return resumedFromStep_; }

  [[nodiscard]] bool WasStopped() const { return stopped_; }

  [[nodiscard]] const TrainingSessionState& State() const { return state_; }

private:
  void Log(const std::string& message) const;

  void Report(const TrainingProgress& progress) const;

  void RestoreOptimizer(Trainer& trainer) const;

  [[nodiscard]] Status Publish() const;

  TrainingJobConfig config_;
  LogFn log_;
  ProgressFn progress_;
  TrainingSessionState state_;
  std::size_t resumedFromStep_ = 0;
  bool stopped_ = false;
};

} // namespace llm::training
