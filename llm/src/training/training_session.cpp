#include "llm/training/training_session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>

#include "llm/model/checkpoint/checkpoint.hpp"
#include "llm/model/transformer.hpp"
#include "llm/quantization/quant_checkpoint.hpp"
#include "llm/tokenizer/bpe_tokenizer.hpp"
#include "llm/training/chat_corpus.hpp"
#include "llm/training/trainer.hpp"
#include "server/json_util.hpp"

namespace llm::training {

namespace fs = std::filesystem;
using server::detail::ExtractBoolField;
using server::detail::ExtractNumberField;
using server::detail::ExtractSizeField;
using server::detail::ExtractStringArrayField;
using server::detail::ExtractStringField;
using server::detail::JsonEscape;
using server::detail::JsonStringArray;

namespace {

constexpr std::size_t kMinAutoSteps = 18000;
constexpr std::size_t kMaxAutoSteps = 30000;
constexpr std::size_t kLogEverySteps = 100;
constexpr std::size_t kReferenceBatchSize = 16; // preset LR'ler bu batch için ayarlı

// nanoGPT: linear warmup → cosine decay to 10% of peak
[[nodiscard]] float ScheduleLearningRate(const std::size_t optStep, const std::size_t totalOptSteps,
                                         const float peakLr) {
  const float minLr = peakLr * 0.1f;
  const std::size_t warmup =
      std::max<std::size_t>(50, std::min<std::size_t>(totalOptSteps / 20, totalOptSteps / 2));
  if (optStep == 0 || totalOptSteps == 0) {
    return minLr;
  }
  if (optStep <= warmup) {
    return peakLr * static_cast<float>(optStep) / static_cast<float>(warmup);
  }
  if (optStep >= totalOptSteps) {
    return minLr;
  }
  const float progress =
      static_cast<float>(optStep - warmup) / static_cast<float>(std::max<std::size_t>(1, totalOptSteps - warmup));
  const float coeff = 0.5f * (1.0f + std::cos(3.14159265358979323846f * progress));
  return minLr + coeff * (peakLr - minLr);
}

[[nodiscard]] std::string NormalizePath(const std::string& path) {
  return fs::path(path).lexically_normal().generic_string();
}

[[nodiscard]] std::vector<std::string> SortedDatasets(const TrainingJobConfig& config) {
  std::vector<std::string> paths;
  paths.reserve(config.datasetPaths.size());
  for (const std::string& path : config.datasetPaths) {
    paths.push_back(NormalizePath(path));
  }
  std::sort(paths.begin(), paths.end());
  paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
  return paths;
}

[[nodiscard]] std::uint64_t FileSizeOrZero(const std::string& path) {
  std::error_code errorCode;
  const auto size = fs::file_size(path, errorCode);
  return errorCode ? 0 : static_cast<std::uint64_t>(size);
}

[[nodiscard]] std::uint64_t FileContentHashOrZero(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    return 0;
  }

  std::uint64_t hash = 14695981039346656037ULL;
  char buffer[8192];
  while (input) {
    input.read(buffer, sizeof(buffer));
    const std::streamsize count = input.gcount();
    for (std::streamsize index = 0; index < count; ++index) {
      hash ^= static_cast<unsigned char>(buffer[index]);
      hash *= 1099511628211ULL;
    }
  }
  return input.bad() ? 0 : hash;
}

[[nodiscard]] std::string Fnv1a64Hex(const std::string& text) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char ch : text) {
    hash ^= ch;
    hash *= 1099511628211ULL;
  }
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << hash;
  return stream.str();
}

[[nodiscard]] std::string FloatText(const float value) {
  std::ostringstream stream;
  stream << std::setprecision(9) << value;
  return stream.str();
}

[[nodiscard]] std::optional<std::string> ReadTextFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    return std::nullopt;
  }
  return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

// Windows refuses to rename over a file another process (antivirus, indexer, a reader)
// still has open; the hold is short-lived, so retry briefly before giving up.
[[nodiscard]] Status ReplaceFile(const std::string& tempPath, const std::string& finalPath) {
  constexpr int kAttempts = 40;
  constexpr auto kDelay = std::chrono::milliseconds(50);
  std::error_code errorCode;
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    fs::rename(tempPath, finalPath, errorCode);
    if (!errorCode) {
      return Status::Ok();
    }
    std::this_thread::sleep_for(kDelay);
  }
  return Status::Fail(ErrorCode::IoError, "cannot replace " + finalPath + ": " + errorCode.message());
}

[[nodiscard]] Status WriteTextFileAtomic(const std::string& path, const std::string& content) {
  const std::string tempPath = path + ".tmp";
  {
    std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
      return Status::Fail(ErrorCode::IoError, "cannot write " + tempPath);
    }
    output << content;
  }
  return ReplaceFile(tempPath, path);
}

[[nodiscard]] bool SameArchitecture(const ModelConfig& lhs, const ModelConfig& rhs) {
  return lhs.numLayers == rhs.numLayers && lhs.hiddenDim == rhs.hiddenDim && lhs.numHeads == rhs.numHeads &&
         lhs.numKvHeads == rhs.numKvHeads && lhs.intermediateDim == rhs.intermediateDim &&
         lhs.maxSeqLen == rhs.maxSeqLen && lhs.ropeTheta == rhs.ropeTheta && lhs.normEps == rhs.normEps;
}

[[nodiscard]] Result<std::vector<DialogueSample>> LoadSamples(const TrainingJobConfig& config) {
  std::vector<DialogueSample> samples = config.includeBuiltin ? DefaultDialogueCorpus() : std::vector<DialogueSample>{};
  for (const std::string& path : SortedDatasets(config)) {
    const auto loaded = LoadDialogueCorpus(path);
    if (!loaded.IsOk()) {
      return Result<std::vector<DialogueSample>>::Fail(loaded.GetError().code,
                                                       "dataset could not be loaded: " + path + " (" +
                                                           loaded.GetError().message + ")");
    }
    samples = MergeDialogueCorpora(std::move(samples), loaded.Value());
  }
  if (samples.empty()) {
    return Result<std::vector<DialogueSample>>::Fail(ErrorCode::InvalidArgument, "no training samples available");
  }
  return Result<std::vector<DialogueSample>>::Ok(std::move(samples));
}

// Deterministic per-epoch batch order so a resumed run replays the same sequence.
class BatchScheduler {
public:
  BatchScheduler(const std::size_t batchCount, const std::uint32_t seed) : batchCount_(batchCount), seed_(seed) {}

  [[nodiscard]] std::size_t IndexFor(const std::size_t step) {
    const std::size_t epoch = step / batchCount_;
    if (epoch != epoch_ || order_.empty()) {
      Rebuild(epoch);
    }
    return order_[step % batchCount_];
  }

private:
  void Rebuild(const std::size_t epoch) {
    epoch_ = epoch;
    order_.resize(batchCount_);
    std::iota(order_.begin(), order_.end(), std::size_t{0});
    if (epoch > 0) {
      std::mt19937 rng(seed_ ^ static_cast<std::uint32_t>(epoch * 0x9E3779B9u));
      std::shuffle(order_.begin(), order_.end(), rng);
    }
  }

  std::size_t batchCount_;
  std::uint32_t seed_;
  std::size_t epoch_ = static_cast<std::size_t>(-1);
  std::vector<std::size_t> order_;
};

[[nodiscard]] Status SaveAtomically(const std::string& path, const std::function<Status(const std::string&)>& writer) {
  const std::string tempPath = path + ".tmp";
  const Status writeStatus = writer(tempPath);
  if (!writeStatus.IsOk()) {
    return writeStatus;
  }
  return ReplaceFile(tempPath, path);
}

[[nodiscard]] Status SaveTempCheckpoint(const model::TransformerModel& model, const AdamW& optimizer,
                                        const TrainingSessionState& state) {
  const std::string& workDir = state.config.workDir;
  const Status modelStatus = SaveAtomically(SessionCheckpointPath(workDir), [&](const std::string& path) {
    return model::Checkpoint::Save(model, path);
  });
  if (!modelStatus.IsOk()) {
    return modelStatus;
  }
  const Status optimizerStatus = SaveAtomically(SessionOptimizerPath(workDir), [&](const std::string& path) {
    return optimizer.SaveState(path);
  });
  if (!optimizerStatus.IsOk()) {
    return optimizerStatus;
  }
  return SaveSessionState(state);
}

} // namespace

TrainingJobConfig DefaultTrainingJobConfig() {
  TrainingJobConfig config;
  config.model = Config::DefaultModelConfig();
  config.model.vocabSize = 3072;
  config.model.hiddenDim = 768;
  config.model.numLayers = 24;
  config.model.numHeads = 12;
  config.model.numKvHeads = 6;
  config.model.intermediateDim = 3072;
  config.model.maxSeqLen = 512;
  // Medium preset varsayılanları
  config.learningRate = 3e-4f;
  config.batchSize = 8;
  return config;
}

std::string ComputeSessionKey(const TrainingJobConfig& config) {
  std::ostringstream canonical;
  for (const std::string& path : SortedDatasets(config)) {
    canonical << path << ':' << FileSizeOrZero(path) << ':' << std::hex << FileContentHashOrZero(path) << std::dec
              << ';';
  }
  const ModelConfig& model = config.model;
  canonical << "builtin=" << config.includeBuiltin << ";layers=" << model.numLayers << ";hidden=" << model.hiddenDim
            << ";heads=" << model.numHeads << ";kv=" << model.numKvHeads << ";inter=" << model.intermediateDim
            << ";seq=" << model.maxSeqLen << ";rope=" << FloatText(model.ropeTheta)
            << ";norm_eps=" << FloatText(model.normEps) << ";vocab=" << model.vocabSize << ";steps=" << config.steps
            << ";batch=" << config.batchSize
            << ";lr=" << FloatText(config.learningRate) << ";seed=" << config.seed;
  return Fnv1a64Hex(canonical.str());
}

std::string SerializeTrainingJobConfigFields(const TrainingJobConfig& config) {
  const ModelConfig& model = config.model;
  std::string json = "\"datasets\":" + JsonStringArray(config.datasetPaths);
  json += std::string(",\"include_builtin\":") + (config.includeBuiltin ? "true" : "false");
  json += ",\"num_layers\":" + std::to_string(model.numLayers);
  json += ",\"hidden_dim\":" + std::to_string(model.hiddenDim);
  json += ",\"num_heads\":" + std::to_string(model.numHeads);
  json += ",\"num_kv_heads\":" + std::to_string(model.numKvHeads);
  json += ",\"intermediate_dim\":" + std::to_string(model.intermediateDim);
  json += ",\"max_seq_len\":" + std::to_string(model.maxSeqLen);
  json += ",\"vocab_size\":" + std::to_string(model.vocabSize);
  json += ",\"steps\":" + std::to_string(config.steps);
  json += ",\"learning_rate\":" + FloatText(config.learningRate);
  json += ",\"checkpoint_interval\":" + std::to_string(config.checkpointInterval);
  json += ",\"batch_size\":" + std::to_string(config.batchSize);
  json += ",\"seed\":" + std::to_string(config.seed);
  json += ",\"work_dir\":\"" + JsonEscape(config.workDir) + "\"";
  json += ",\"output_model\":\"" + JsonEscape(config.outputModelPath) + "\"";
  json += ",\"output_quant\":\"" + JsonEscape(config.outputQuantPath) + "\"";
  json += ",\"output_tokenizer\":\"" + JsonEscape(config.outputTokenizerPath) + "\"";
  return json;
}

TrainingJobConfig ParseTrainingJobConfig(const std::string& json, const TrainingJobConfig& defaults) {
  TrainingJobConfig config = defaults;
  const auto sizeOr = [&json](const char* key, std::size_t& target) {
    if (const auto value = ExtractSizeField(json, key)) {
      target = *value;
    }
  };
  const auto stringOr = [&json](const char* key, std::string& target) {
    if (const auto value = ExtractStringField(json, key)) {
      target = *value;
    }
  };

  if (const auto datasets = ExtractStringArrayField(json, "datasets")) {
    config.datasetPaths = *datasets;
  }
  if (const auto builtin = ExtractBoolField(json, "include_builtin")) {
    config.includeBuiltin = *builtin;
  }
  sizeOr("num_layers", config.model.numLayers);
  sizeOr("hidden_dim", config.model.hiddenDim);
  sizeOr("num_heads", config.model.numHeads);
  sizeOr("num_kv_heads", config.model.numKvHeads);
  sizeOr("intermediate_dim", config.model.intermediateDim);
  sizeOr("max_seq_len", config.model.maxSeqLen);
  sizeOr("vocab_size", config.model.vocabSize);
  sizeOr("steps", config.steps);
  sizeOr("checkpoint_interval", config.checkpointInterval);
  sizeOr("batch_size", config.batchSize);
  if (const auto rate = ExtractNumberField(json, "learning_rate")) {
    config.learningRate = *rate;
  }
  if (const auto seed = ExtractSizeField(json, "seed")) {
    config.seed = static_cast<std::uint32_t>(*seed);
  }
  stringOr("work_dir", config.workDir);
  stringOr("output_model", config.outputModelPath);
  stringOr("output_quant", config.outputQuantPath);
  stringOr("output_tokenizer", config.outputTokenizerPath);
  return config;
}

std::string SerializeSessionState(const TrainingSessionState& state) {
  return "{\"session_key\":\"" + JsonEscape(state.sessionKey) + "\",\"completed_steps\":" +
         std::to_string(state.completedSteps) + ",\"total_steps\":" + std::to_string(state.totalSteps) +
         ",\"last_loss\":" + FloatText(state.lastLoss) + ",\"sample_count\":" + std::to_string(state.sampleCount) +
         "," + SerializeTrainingJobConfigFields(state.config) + "}";
}

std::optional<TrainingSessionState> ParseSessionState(const std::string& json) {
  const auto key = ExtractStringField(json, "session_key");
  if (!key.has_value()) {
    return std::nullopt;
  }
  TrainingSessionState state;
  state.sessionKey = *key;
  state.completedSteps = ExtractSizeField(json, "completed_steps").value_or(0);
  state.totalSteps = ExtractSizeField(json, "total_steps").value_or(0);
  state.lastLoss = ExtractNumberField(json, "last_loss").value_or(0.0f);
  state.sampleCount = ExtractSizeField(json, "sample_count").value_or(0);
  state.config = ParseTrainingJobConfig(json, DefaultTrainingJobConfig());
  return state;
}

std::string SessionStatePath(const std::string& workDir) {
  return (fs::path(workDir) / "session.json").string();
}

std::string SessionCheckpointPath(const std::string& workDir) {
  return (fs::path(workDir) / "model.ckpt").string();
}

std::string SessionTokenizerPath(const std::string& workDir) {
  return (fs::path(workDir) / "tokenizer").string();
}

std::string SessionOptimizerPath(const std::string& workDir) {
  return (fs::path(workDir) / "optimizer.bin").string();
}

std::optional<TrainingSessionState> LoadSessionState(const std::string& workDir) {
  const auto content = ReadTextFile(SessionStatePath(workDir));
  return content ? ParseSessionState(*content) : std::nullopt;
}

Status SaveSessionState(const TrainingSessionState& state) {
  std::error_code errorCode;
  fs::create_directories(state.config.workDir, errorCode);
  return WriteTextFileAtomic(SessionStatePath(state.config.workDir), SerializeSessionState(state));
}

void ClearWorkDir(const std::string& workDir) {
  std::error_code errorCode;
  fs::remove_all(workDir, errorCode);
}

std::size_t ResolveTrainingSteps(const std::size_t requestedSteps, const std::size_t sampleCount) {
  if (requestedSteps > 0) {
    return requestedSteps;
  }
  return std::min(kMaxAutoSteps, std::max(kMinAutoSteps, sampleCount * 2));
}

float ResolveLearningRate(const float requestedRate, const std::size_t sampleCount) {
  if (requestedRate > 0.0f) {
    return requestedRate;
  }
  return sampleCount > 1000 ? 3e-4f : 5e-4f;
}

TrainingRunner::TrainingRunner(TrainingJobConfig config, LogFn log, ProgressFn progress)
    : config_(std::move(config)), log_(std::move(log)), progress_(std::move(progress)) {
  if (config_.checkpointInterval == 0) {
    config_.checkpointInterval = 1;
  }
}

void TrainingRunner::Log(const std::string& message) const {
  if (log_) {
    log_(message);
  }
}

void TrainingRunner::Report(const TrainingProgress& progress) const {
  if (progress_) {
    progress_(progress);
  }
}

void TrainingRunner::RestoreOptimizer(Trainer& trainer) const {
  const std::string path = SessionOptimizerPath(config_.workDir);
  const Status status = trainer.Optimizer().LoadState(path, trainer.Parameters());
  if (status.IsOk()) {
    Log("Optimizer state restored (" + std::to_string(trainer.Optimizer().StepCount()) + " steps)");
  } else {
    Log("Optimizer state unavailable (" + status.GetError().message + "); moments reset");
  }
}

Status TrainingRunner::Run(const std::atomic<bool>& stopRequested) {
  const Status configStatus = config_.model.Validate();
  if (!configStatus.IsOk()) {
    return configStatus;
  }

  const auto samplesResult = LoadSamples(config_);
  if (!samplesResult.IsOk()) {
    return Status::Fail(samplesResult.GetError().code, samplesResult.GetError().message);
  }
  const std::vector<DialogueSample>& samples = samplesResult.Value();
  Log("Samples: " + std::to_string(samples.size()) + " unique (" + std::to_string(config_.datasetPaths.size()) +
      " dataset file(s)" + (config_.includeBuiltin ? ", built-in dialogues included" : "") + ")");

  const std::string sessionKey = ComputeSessionKey(config_);
  const std::string checkpointPath = SessionCheckpointPath(config_.workDir);
  const std::string tokenizerPath = SessionTokenizerPath(config_.workDir);
  const auto previous = LoadSessionState(config_.workDir);
  bool resuming = previous.has_value() && previous->sessionKey == sessionKey &&
                  previous->completedSteps < previous->totalSteps && fs::exists(checkpointPath) &&
                  fs::exists(tokenizerPath);
  if (previous.has_value() && !resuming) {
    Log("Previous session does not match this configuration; clearing temp checkpoint");
  }

  BpeTokenizer tokenizer;
  if (resuming) {
    auto loaded = BpeTokenizer::Load(tokenizerPath);
    if (loaded.IsOk()) {
      tokenizer = std::move(loaded.Value());
    } else {
      Log("Temp tokenizer unreadable; starting fresh");
      resuming = false;
    }
  }

  std::optional<model::TransformerModel> model;
  if (resuming) {
    auto loaded = model::Checkpoint::Load(checkpointPath);
    if (loaded.IsOk() && SameArchitecture(loaded.Value().GetConfig(), config_.model)) {
      model.emplace(std::move(loaded.Value()));
    } else {
      Log("Temp checkpoint unreadable or incompatible; starting fresh");
      resuming = false;
    }
  }

  if (!resuming) {
    ClearWorkDir(config_.workDir);
    const std::string corpus = BuildTokenizerCorpus(samples);
    Log("Training tokenizer on " + std::to_string(corpus.size() / (1024 * 1024)) + " MB corpus (target vocab " +
        std::to_string(config_.model.vocabSize) + ")");
    const auto tokenizerProgress = [this](const std::size_t vocab, const std::size_t target) {
      Log("Tokenizer vocab " + std::to_string(vocab) + "/" + std::to_string(target));
    };
    if (!tokenizer.Train(corpus, config_.model.vocabSize, tokenizerProgress).IsOk()) {
      return Status::Fail(ErrorCode::Internal, "tokenizer training failed");
    }
    Log("Tokenizer ready (" + std::to_string(tokenizer.GetVocabulary().Size()) + " tokens)");
    std::error_code errorCode;
    fs::create_directories(tokenizerPath, errorCode);
    if (!tokenizer.Save(tokenizerPath).IsOk()) {
      return Status::Fail(ErrorCode::IoError, "tokenizer could not be saved to " + tokenizerPath);
    }

    ModelConfig modelConfig = config_.model;
    modelConfig.vocabSize = tokenizer.GetVocabulary().Size();
    model.emplace(modelConfig);
    std::mt19937 rng(config_.seed);
    model->ResetParameters(rng);

    state_ = TrainingSessionState{};
    state_.sessionKey = sessionKey;
    state_.totalSteps = ResolveTrainingSteps(config_.steps, samples.size());
    state_.sampleCount = samples.size();
    state_.config = config_;
  } else {
    state_ = *previous;
    state_.config = config_;
    resumedFromStep_ = state_.completedSteps;
    Log("Resuming session at step " + std::to_string(resumedFromStep_) + "/" + std::to_string(state_.totalSteps));
  }

  if (stopRequested.load()) {
    stopped_ = true;
    Log("Stopped before training started");
    return Status::Ok();
  }

  const ModelConfig& modelConfig = model->GetConfig();
  Log("Tokenizing " + std::to_string(samples.size()) + " samples");
  std::vector<std::vector<TokenId>> batches =
      BuildTrainingBatches(tokenizer, samples, modelConfig.maxSeqLen, [this](const std::size_t done, const std::size_t total) {
        Log("Tokenized " + std::to_string(done) + "/" + std::to_string(total));
      });
  if (batches.empty()) {
    return Status::Fail(ErrorCode::InvalidArgument, "no valid training batches after tokenization");
  }

  TrainerConfig trainerConfig;
  const float configuredLr = ResolveLearningRate(config_.learningRate, samples.size());
  const std::size_t batchSize = std::max(config_.batchSize, std::size_t(1));
  // Batch 1 ile preset LR (batch 16 için) kullanılırsa loss warmup sonrası patlar.
  // sqrt scaling: batch küçülünce LR orantılı düşer.
  const float peakLearningRate =
      configuredLr * std::sqrt(static_cast<float>(batchSize) / static_cast<float>(kReferenceBatchSize));
  trainerConfig.optimizer.learningRate = peakLearningRate;
  Trainer trainer(*model, trainerConfig);
  if (resuming) {
    RestoreOptimizer(trainer);
  }
  const std::size_t totalOptSteps =
      std::max<std::size_t>(1, (state_.totalSteps + batchSize - 1) / batchSize);
  Log("Training: batches=" + std::to_string(batches.size()) + " steps=" + std::to_string(state_.totalSteps) +
      " vocab=" + std::to_string(modelConfig.vocabSize) + " layers=" + std::to_string(modelConfig.numLayers) +
      " hidden=" + std::to_string(modelConfig.hiddenDim) + " batch_size=" + std::to_string(batchSize) +
      " peak_lr=" + FloatText(peakLearningRate) + " (configured=" + FloatText(configuredLr) + ")" +
      " opt_steps≈" + std::to_string(totalOptSteps) +
      " checkpoint_every=" + std::to_string(config_.checkpointInterval));

  BatchScheduler scheduler(batches.size(), config_.seed);
  std::size_t lastSavedStep = state_.completedSteps;

  // Each iteration processes one mini-batch (batchSize samples)
  // state_.completedSteps tracks number of SAMPLES processed, not weight updates
  while (state_.completedSteps < state_.totalSteps) {
    if (stopRequested.load()) {
      stopped_ = true;
      break;
    }

    // Accumulate gradients for mini-batch
    float batchLoss = 0.0f;
    std::size_t samplesProcessed = 0;
    bool firstInBatch = true;

    for (std::size_t i = 0; i < batchSize && state_.completedSteps < state_.totalSteps; ++i) {
      const std::size_t sampleIdx = state_.completedSteps;
      float sampleLoss = 0.0f;

      // Use AccumulateGradients: first sample zeros grad, subsequent ones accumulate
      const Status status = trainer.AccumulateGradients(
          batches[scheduler.IndexFor(sampleIdx)], sampleLoss, firstInBatch);
      if (!status.IsOk()) {
        return status;
      }

      batchLoss += sampleLoss;
      samplesProcessed++;
      state_.completedSteps++;
      firstInBatch = false;
    }

    if (samplesProcessed == 0) {
      break;
    }

    // Mini-batch gradientlerini normalize et ve clip'le
    ParameterList& params = trainer.Parameters();
    constexpr float kGradientMaxNorm = 1.0f;

    // 1. Gradientleri batch boyutuna böl (ortalama al)
    params.ScaleGradients(1.0f / static_cast<float>(samplesProcessed));

    // 2. Gradient clipping (exploding gradient'i önle)
    const float gradNormBefore = params.GradientNorm();
    params.ClipGradients(kGradientMaxNorm);

    // nanoGPT schedule: warmup + cosine decay (fixed 200-step cliff yok)
    const std::size_t nextOptStep = trainer.Optimizer().StepCount() + 1;
    const float scheduledLr = ScheduleLearningRate(nextOptStep, totalOptSteps, peakLearningRate);
    trainer.Optimizer().SetLearningRate(scheduledLr);

    // Apply one optimizer step for the entire mini-batch
    const Status stepStatus = trainer.ApplyStep();
    if (!stepStatus.IsOk()) {
      return stepStatus;
    }

    batchLoss /= static_cast<float>(samplesProcessed);
    state_.lastLoss = batchLoss;

    // Checkpoint: check if we crossed a checkpoint boundary in sample count
    const bool checkpoint =
        (config_.checkpointInterval > 0 &&
         (lastSavedStep / config_.checkpointInterval) < (state_.completedSteps / config_.checkpointInterval)) ||
        state_.completedSteps == state_.totalSteps;

    if (checkpoint) {
      const Status saveStatus = SaveTempCheckpoint(*model, trainer.Optimizer(), state_);
      if (!saveStatus.IsOk()) {
        return saveStatus;
      }
      lastSavedStep = state_.completedSteps;
    }

    Report(TrainingProgress{.step = state_.completedSteps, .totalSteps = state_.totalSteps, .loss = batchLoss,
                            .checkpointSaved = checkpoint});

    // Log every kLogEverySteps samples
    const std::size_t prevLogBoundary = (state_.completedSteps - samplesProcessed) / kLogEverySteps;
    const std::size_t currLogBoundary = state_.completedSteps / kLogEverySteps;
    if (currLogBoundary > prevLogBoundary || state_.completedSteps == state_.totalSteps) {
      Log("step " + std::to_string(state_.completedSteps) + "/" + std::to_string(state_.totalSteps) +
          " loss=" + FloatText(batchLoss) +
          " lr=" + FloatText(scheduledLr) +
          (gradNormBefore > kGradientMaxNorm ? " (grad clipped: " + FloatText(gradNormBefore) + " -> " + FloatText(kGradientMaxNorm) + ")" : ""));
    }
  }

  if (stopped_) {
    if (lastSavedStep != state_.completedSteps) {
      const Status saveStatus = SaveTempCheckpoint(*model, trainer.Optimizer(), state_);
      if (!saveStatus.IsOk()) {
        return saveStatus;
      }
    }
    Log("Stopped at step " + std::to_string(state_.completedSteps) + "/" + std::to_string(state_.totalSteps) +
        "; temp checkpoint kept for resume");
    return Status::Ok();
  }

  return Publish();
}

Status TrainingRunner::Publish() const {
  std::error_code errorCode;
  const std::string modelTemp = config_.outputModelPath + ".tmp";
  fs::copy_file(SessionCheckpointPath(config_.workDir), modelTemp, fs::copy_options::overwrite_existing, errorCode);
  if (errorCode) {
    return Status::Fail(ErrorCode::IoError, "cannot copy final checkpoint: " + errorCode.message());
  }
  const Status modelStatus = ReplaceFile(modelTemp, config_.outputModelPath);
  if (!modelStatus.IsOk()) {
    return modelStatus;
  }

  const std::string quantTemp = config_.outputQuantPath + ".tmp";
  const Status quantStatus = quantization::QuantCheckpoint::ConvertFile(config_.outputModelPath, quantTemp);
  if (!quantStatus.IsOk()) {
    return quantStatus;
  }
  const Status quantReplace = ReplaceFile(quantTemp, config_.outputQuantPath);
  if (!quantReplace.IsOk()) {
    return quantReplace;
  }

  fs::create_directories(config_.outputTokenizerPath, errorCode);
  fs::copy(SessionTokenizerPath(config_.workDir), config_.outputTokenizerPath,
           fs::copy_options::overwrite_existing | fs::copy_options::recursive, errorCode);
  if (errorCode) {
    return Status::Fail(ErrorCode::IoError, "cannot publish tokenizer: " + errorCode.message());
  }

  ClearWorkDir(config_.workDir);
  Log("Training complete. Published " + config_.outputModelPath + ", " + config_.outputQuantPath + " and " +
      config_.outputTokenizerPath);
  return Status::Ok();
}

} // namespace llm::training
