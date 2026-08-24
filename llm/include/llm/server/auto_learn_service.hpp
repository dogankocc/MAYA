#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "llm/core/status.hpp"
#include "llm/training/fine_tune.hpp"
#include "llm/training/jsonl_writer.hpp"

namespace llm::server {

struct AutoLearnConfig {
  bool enabled = true;
  std::string corpusPath = "data/corpus/auto_learn.jsonl";
  std::string statePath = "data/corpus/auto_learn_state.json";
  std::string fp32ModelPath = "model.ckpt";
  std::string quantModelPath = "model.ckptq";
  std::string tokenizerPath = "tokenizer_data";
  std::size_t samplesBeforeTrain = 6;
  std::size_t minResponseChars = 8;
  training::FineTuneOptions fineTune;
};

struct AutoLearnStatus {
  bool enabled = false;
  bool training = false;
  std::size_t pendingSamples = 0;
  std::size_t totalSaved = 0;
  std::int64_t lastTrainUnix = 0;
  std::string lastError;
};

class AutoLearnService {
public:
  using ReloadModelCallback = std::function<Status()>;

  AutoLearnService(AutoLearnConfig config, ReloadModelCallback reloadModel);

  void OnChatCompleted(const std::string& prompt, const std::string& response, const std::string& intent);

  [[nodiscard]] AutoLearnStatus GetStatus() const;

  void Shutdown();

private:
  [[nodiscard]] std::filesystem::path ResolveInputCheckpoint() const;

  void EnsureParentDirectory(const std::filesystem::path& path) const;

  void LoadStateLocked();

  void SaveStateLocked();

  void MaybeStartTrainingLocked();

  void RunTrainingJob();

  AutoLearnConfig config_;
  ReloadModelCallback reloadModel_;
  mutable std::mutex mutex_;
  std::size_t pendingSamples_ = 0;
  std::size_t totalSaved_ = 0;
  std::int64_t lastTrainUnix_ = 0;
  std::string lastError_;
  std::atomic<bool> training_{false};
  std::atomic<bool> shutdown_{false};
  std::thread worker_;
};

} // namespace llm::server
