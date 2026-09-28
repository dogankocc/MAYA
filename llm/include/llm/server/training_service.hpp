#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/training/training_session.hpp"

namespace llm::server {

enum class TrainingJobState { Idle, Running, Stopping, Completed, Stopped, Failed };

[[nodiscard]] const char* TrainingJobStateName(TrainingJobState state);

struct TrainingServiceConfig {
  std::string datasetsDir = "data/corpus";
  std::vector<std::string> extraDatasetFiles = {"data/chat_corpus_tr.jsonl"};
  std::string workDir = "training_work";
  std::string chatServerUrl = "http://127.0.0.1:8765";
  std::string pythonCommand = "python";
  std::size_t maxLogEntries = 2000;
};

struct TrainingStatusSnapshot {
  TrainingJobState state = TrainingJobState::Idle;
  std::size_t step = 0;
  std::size_t totalSteps = 0;
  float loss = 0.0f;
  std::size_t resumedFromStep = 0;
  double elapsedSeconds = 0.0;
  double stepsPerSecond = 0.0;
  std::string message;
  bool downloading = false;
  std::optional<training::TrainingJobConfig> activeConfig;
};

struct DatasetInfo {
  std::string name;
  std::string path;
  std::string description;
  std::uint64_t sizeBytes = 0;
  std::size_t sampleCount = 0;
};

struct StorageInfo {
  std::uint64_t freeBytes = 0;
  std::uint64_t totalBytes = 0;
  std::uint64_t modelBytes = 0;
  std::uint64_t quantBytes = 0;
  std::uint64_t tokenizerBytes = 0;
  std::uint64_t tempCheckpointBytes = 0;
};

struct LogEntry {
  std::uint64_t seq = 0;
  std::int64_t unixTime = 0;
  std::string text;
};

// Owns a single background training job (one at a time) plus dataset download
// helpers. Thread-safe; designed to be polled by the HTTP layer.
class TrainingService {
public:
  explicit TrainingService(TrainingServiceConfig config = {});
  ~TrainingService();

  TrainingService(const TrainingService&) = delete;
  TrainingService& operator=(const TrainingService&) = delete;

  [[nodiscard]] Status Start(training::TrainingJobConfig jobConfig);

  void RequestStop();

  [[nodiscard]] TrainingStatusSnapshot GetStatus() const;

  [[nodiscard]] std::vector<LogEntry> GetLogs(std::uint64_t sinceSeq, std::size_t limit = 500) const;

  [[nodiscard]] std::vector<DatasetInfo> ListDatasets() const;

  [[nodiscard]] StorageInfo GetStorage() const;

  [[nodiscard]] std::optional<training::TrainingSessionState> PendingSession() const;

  [[nodiscard]] Status ClearPendingSession();

  // scriptName: "code" | "turkish"
  [[nodiscard]] Status StartDownload(const std::string& scriptName);

  [[nodiscard]] const TrainingServiceConfig& Config() const { return config_; }

  void Shutdown();

private:
  void AppendLog(const std::string& text);

  void RunJob(training::TrainingJobConfig jobConfig);

  void RunDownload(std::string command);

  void NotifyChatServerReload();

  [[nodiscard]] bool IsTrainingActive() const;

  void JoinFinishedWorkers();

  TrainingServiceConfig config_;
  std::mutex lifecycleMutex_;
  mutable std::mutex mutex_;
  std::thread worker_;
  std::thread downloadWorker_;
  std::atomic<bool> stopRequested_{false};
  std::atomic<bool> downloading_{false};
  TrainingStatusSnapshot status_;
  std::chrono::steady_clock::time_point startedAt_;
  std::deque<LogEntry> logs_;
  std::uint64_t nextLogSeq_ = 1;
};

} // namespace llm::server
