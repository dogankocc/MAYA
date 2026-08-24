#include "llm/server/auto_learn_service.hpp"

#include <chrono>
#include <iostream>

#include "llm/training/dialogue_format.hpp"

namespace llm::server {

namespace {

[[nodiscard]] std::int64_t NowUnix() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

} // namespace

AutoLearnService::AutoLearnService(AutoLearnConfig config, ReloadModelCallback reloadModel)
    : config_(std::move(config)), reloadModel_(std::move(reloadModel)) {
  config_.fineTune.corpusPath = config_.corpusPath;
  config_.fineTune.fp32OutputPath = config_.fp32ModelPath;
  config_.fineTune.quantOutputPath = config_.quantModelPath;
  config_.fineTune.tokenizerPath = config_.tokenizerPath;

  std::lock_guard<std::mutex> lock(mutex_);
  LoadStateLocked();
}

void AutoLearnService::EnsureParentDirectory(const std::filesystem::path& path) const {
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

void AutoLearnService::LoadStateLocked() {
  pendingSamples_ = 0;
  totalSaved_ = 0;
  lastTrainUnix_ = 0;

  std::ifstream input(config_.statePath);
  if (!input.is_open()) {
    return;
  }

  std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  const auto readNumber = [&content](const std::string& key) -> std::size_t {
    const std::string pattern = "\"" + key + "\":";
    const std::size_t pos = content.find(pattern);
    if (pos == std::string::npos) {
      return 0;
    }
    return static_cast<std::size_t>(std::stoull(content.substr(pos + pattern.size())));
  };

  pendingSamples_ = readNumber("pending");
  totalSaved_ = readNumber("total_saved");
  lastTrainUnix_ = static_cast<std::int64_t>(readNumber("last_train_unix"));
}

void AutoLearnService::SaveStateLocked() {
  EnsureParentDirectory(std::filesystem::path(config_.statePath));
  std::ofstream output(config_.statePath, std::ios::trunc);
  if (!output.is_open()) {
    return;
  }

  output << "{\"pending\":" << pendingSamples_ << ",\"total_saved\":" << totalSaved_
         << ",\"last_train_unix\":" << lastTrainUnix_ << "}";
}

void AutoLearnService::OnChatCompleted(const std::string& prompt, const std::string& response,
                                       const std::string& intent) {
  if (!config_.enabled || shutdown_.load()) {
    return;
  }

  if (prompt.empty() || response.size() < config_.minResponseChars) {
    return;
  }

  if (training::IsLowQualityResponse(response) || training::IsFallbackResponse(response)) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureParentDirectory(std::filesystem::path(config_.corpusPath));
    std::ofstream corpus(config_.corpusPath, std::ios::app);
    if (!corpus.is_open()) {
      lastError_ = "failed to append auto-learn corpus";
      return;
    }

    corpus << training::FormatDialogueJsonlLine(intent, prompt, response);
    pendingSamples_ += 1;
    totalSaved_ += 1;
    lastError_.clear();
    SaveStateLocked();

    std::cout << "[auto-learn] saved sample pending=" << pendingSamples_ << " total=" << totalSaved_ << '\n';
    MaybeStartTrainingLocked();
  }
}

std::filesystem::path AutoLearnService::ResolveInputCheckpoint() const {
  if (std::filesystem::exists(config_.fp32ModelPath)) {
    return config_.fp32ModelPath;
  }
  if (std::filesystem::exists(config_.quantModelPath)) {
    return config_.quantModelPath;
  }
  return {};
}

void AutoLearnService::MaybeStartTrainingLocked() {
  if (training_.load() || pendingSamples_ < config_.samplesBeforeTrain) {
    return;
  }

  if (ResolveInputCheckpoint().empty()) {
    lastError_ = "waiting for base model before auto-learn training";
    return;
  }

  if (worker_.joinable()) {
    worker_.join();
  }

  training_.store(true);
  worker_ = std::thread([this]() { RunTrainingJob(); });
}

void AutoLearnService::RunTrainingJob() {
  training::FineTuneOptions options = config_.fineTune;
  options.inputCheckpointPath = ResolveInputCheckpoint().string();

  const Status trainStatus = training::RunFineTuneOnCorpus(options);
  if (!trainStatus.IsOk()) {
    std::lock_guard<std::mutex> lock(mutex_);
    lastError_ = trainStatus.Message();
    training_.store(false);
    std::cerr << "[auto-learn] training failed: " << lastError_ << '\n';
    return;
  }

  if (reloadModel_) {
    const Status reloadStatus = reloadModel_();
    if (!reloadStatus.IsOk()) {
      std::lock_guard<std::mutex> lock(mutex_);
      lastError_ = reloadStatus.Message();
      training_.store(false);
      std::cerr << "[auto-learn] reload failed: " << lastError_ << '\n';
      return;
    }
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    pendingSamples_ = 0;
    lastTrainUnix_ = NowUnix();
    lastError_.clear();
    SaveStateLocked();
  }

  training_.store(false);
  std::cout << "[auto-learn] training complete, model reloaded\n";
}

AutoLearnStatus AutoLearnService::GetStatus() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return AutoLearnStatus{
      .enabled = config_.enabled,
      .training = training_.load(),
      .pendingSamples = pendingSamples_,
      .totalSaved = totalSaved_,
      .lastTrainUnix = lastTrainUnix_,
      .lastError = lastError_,
  };
}

void AutoLearnService::Shutdown() {
  shutdown_.store(true);
  if (worker_.joinable()) {
    worker_.join();
  }
}

} // namespace llm::server
