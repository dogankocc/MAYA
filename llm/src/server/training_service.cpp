#include "llm/server/training_service.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "httplib.h"

#ifdef _WIN32
#define LLM_POPEN _popen
#define LLM_PCLOSE _pclose
#else
#define LLM_POPEN popen
#define LLM_PCLOSE pclose
#endif

namespace llm::server {

namespace fs = std::filesystem;

namespace {

[[nodiscard]] std::int64_t NowUnix() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

[[nodiscard]] std::uint64_t FileSizeOrZero(const fs::path& path) {
  std::error_code errorCode;
  const auto size = fs::file_size(path, errorCode);
  return errorCode ? 0 : static_cast<std::uint64_t>(size);
}

[[nodiscard]] std::uint64_t DirectorySize(const fs::path& path) {
  std::error_code errorCode;
  if (!fs::is_directory(path, errorCode)) {
    return FileSizeOrZero(path);
  }
  std::uint64_t total = 0;
  for (const auto& entry : fs::recursive_directory_iterator(path, errorCode)) {
    if (entry.is_regular_file(errorCode)) {
      total += FileSizeOrZero(entry.path());
    }
  }
  return total;
}

[[nodiscard]] std::string TrimCopy(std::string value) {
  const auto notSpace = [](const unsigned char ch) { return !std::isspace(ch); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
  value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
  return value;
}

[[nodiscard]] DatasetInfo DescribeDataset(const fs::path& path) {
  DatasetInfo info;
  info.name = path.filename().string();
  info.path = path.lexically_normal().generic_string();
  info.sizeBytes = FileSizeOrZero(path);

  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line)) {
    line = TrimCopy(line);
    if (line.empty()) {
      continue;
    }
    if (line[0] == '#') {
      if (info.description.empty()) {
        info.description = TrimCopy(line.substr(1));
      }
      continue;
    }
    info.sampleCount += 1;
  }
  return info;
}

[[nodiscard]] std::optional<std::string> ResolveDownloadScript(const std::string& scriptName) {
  if (scriptName == "code") {
    return "scripts/download_code_dataset.py";
  }
  if (scriptName == "turkish") {
    return "scripts/download_training_corpus.py";
  }
  return std::nullopt;
}

} // namespace

const char* TrainingJobStateName(const TrainingJobState state) {
  switch (state) {
  case TrainingJobState::Idle:
    return "idle";
  case TrainingJobState::Running:
    return "running";
  case TrainingJobState::Stopping:
    return "stopping";
  case TrainingJobState::Completed:
    return "completed";
  case TrainingJobState::Stopped:
    return "stopped";
  case TrainingJobState::Failed:
    return "failed";
  }
  return "unknown";
}

TrainingService::TrainingService(TrainingServiceConfig config) : config_(std::move(config)) {}

TrainingService::~TrainingService() {
  Shutdown();
}

void TrainingService::AppendLog(const std::string& text) {
  std::lock_guard<std::mutex> lock(mutex_);
  logs_.push_back(LogEntry{.seq = nextLogSeq_++, .unixTime = NowUnix(), .text = text});
  while (logs_.size() > config_.maxLogEntries) {
    logs_.pop_front();
  }
}

bool TrainingService::IsTrainingActive() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_.state == TrainingJobState::Running || status_.state == TrainingJobState::Stopping;
}

// Must be called with lifecycleMutex_ held and mutex_ released (workers log while finishing).
void TrainingService::JoinFinishedWorkers() {
  if (worker_.joinable() && !IsTrainingActive()) {
    worker_.join();
  }
  if (downloadWorker_.joinable() && !downloading_.load()) {
    downloadWorker_.join();
  }
}

Status TrainingService::Start(training::TrainingJobConfig jobConfig) {
  std::lock_guard<std::mutex> lifecycle(lifecycleMutex_);
  if (IsTrainingActive()) {
    return Status::Fail(ErrorCode::InvalidArgument, "training is already running");
  }
  if (downloading_.load()) {
    return Status::Fail(ErrorCode::InvalidArgument, "dataset download in progress");
  }
  JoinFinishedWorkers();

  jobConfig.workDir = config_.workDir;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = TrainingStatusSnapshot{};
    status_.state = TrainingJobState::Running;
    status_.activeConfig = jobConfig;
    stopRequested_.store(false);
    startedAt_ = std::chrono::steady_clock::now();
  }
  worker_ = std::thread([this, jobConfig]() { RunJob(jobConfig); });
  return Status::Ok();
}

void TrainingService::RequestStop() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (status_.state != TrainingJobState::Running) {
    return;
  }
  status_.state = TrainingJobState::Stopping;
  stopRequested_.store(true);
}

void TrainingService::RunJob(training::TrainingJobConfig jobConfig) {
  AppendLog("Training job started");

  const auto onProgress = [this](const training::TrainingProgress& progress) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (status_.step == 0 && progress.step > 0) {
      status_.resumedFromStep = progress.step - 1;
    }
    status_.step = progress.step;
    status_.totalSteps = progress.totalSteps;
    status_.loss = progress.loss;
  };

  training::TrainingRunner runner(std::move(jobConfig), [this](const std::string& text) { AppendLog(text); },
                                  onProgress);
  const Status result = runner.Run(stopRequested_);

  bool completed = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.elapsedSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt_).count();
    if (!result.IsOk()) {
      status_.state = TrainingJobState::Failed;
      status_.message = result.Message();
    } else if (runner.WasStopped()) {
      status_.state = TrainingJobState::Stopped;
      status_.message = "stopped at step " + std::to_string(runner.State().completedSteps);
    } else {
      status_.state = TrainingJobState::Completed;
      status_.message = "model published";
      completed = true;
    }
  }

  if (!result.IsOk()) {
    AppendLog("ERROR: " + result.Message());
  }
  if (completed) {
    NotifyChatServerReload();
  }
}

void TrainingService::NotifyChatServerReload() {
  httplib::Client client(config_.chatServerUrl);
  client.set_connection_timeout(2, 0);
  client.set_read_timeout(60, 0);
  const auto response = client.Post("/api/v1/reload");
  if (response && response->status == 200) {
    AppendLog("Chat server reloaded the new model");
  } else {
    AppendLog("Chat server reload skipped (server not reachable); restart it to load the new model");
  }
}

TrainingStatusSnapshot TrainingService::GetStatus() const {
  std::lock_guard<std::mutex> lock(mutex_);
  TrainingStatusSnapshot snapshot = status_;
  snapshot.downloading = downloading_.load();
  if (snapshot.state == TrainingJobState::Running || snapshot.state == TrainingJobState::Stopping) {
    snapshot.elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt_).count();
  }
  if (snapshot.elapsedSeconds > 0.0 && snapshot.step > snapshot.resumedFromStep) {
    snapshot.stepsPerSecond = static_cast<double>(snapshot.step - snapshot.resumedFromStep) / snapshot.elapsedSeconds;
  }
  return snapshot;
}

std::vector<LogEntry> TrainingService::GetLogs(const std::uint64_t sinceSeq, const std::size_t limit) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<LogEntry> entries;
  for (const LogEntry& entry : logs_) {
    if (entry.seq > sinceSeq) {
      entries.push_back(entry);
      if (entries.size() >= limit) {
        break;
      }
    }
  }
  return entries;
}

std::vector<DatasetInfo> TrainingService::ListDatasets() const {
  std::vector<fs::path> files;
  std::error_code errorCode;
  for (const auto& entry : fs::directory_iterator(config_.datasetsDir, errorCode)) {
    if (entry.is_regular_file(errorCode) && entry.path().extension() == ".jsonl") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  for (const std::string& extra : config_.extraDatasetFiles) {
    if (fs::is_regular_file(extra, errorCode)) {
      files.emplace_back(extra);
    }
  }

  std::vector<DatasetInfo> datasets;
  datasets.reserve(files.size());
  for (const fs::path& file : files) {
    datasets.push_back(DescribeDataset(file));
  }
  return datasets;
}

StorageInfo TrainingService::GetStorage() const {
  const training::TrainingJobConfig defaults = training::DefaultTrainingJobConfig();
  StorageInfo info;
  std::error_code errorCode;
  const fs::space_info space = fs::space(".", errorCode);
  if (!errorCode) {
    info.freeBytes = space.available;
    info.totalBytes = space.capacity;
  }
  info.modelBytes = FileSizeOrZero(defaults.outputModelPath);
  info.quantBytes = FileSizeOrZero(defaults.outputQuantPath);
  info.tokenizerBytes = DirectorySize(defaults.outputTokenizerPath);
  info.tempCheckpointBytes = DirectorySize(config_.workDir);
  return info;
}

std::optional<training::TrainingSessionState> TrainingService::PendingSession() const {
  // The worker rewrites session.json on every checkpoint; reading it concurrently
  // would hold a handle open and make the rename fail on Windows.
  if (IsTrainingActive()) {
    return std::nullopt;
  }
  auto state = training::LoadSessionState(config_.workDir);
  if (!state.has_value() || state->completedSteps >= state->totalSteps ||
      !fs::exists(training::SessionCheckpointPath(config_.workDir))) {
    return std::nullopt;
  }
  return state;
}

Status TrainingService::ClearPendingSession() {
  if (IsTrainingActive()) {
    return Status::Fail(ErrorCode::InvalidArgument, "stop the running training first");
  }
  training::ClearWorkDir(config_.workDir);
  AppendLog("Temp training session cleared");
  return Status::Ok();
}

Status TrainingService::StartDownload(const std::string& scriptName) {
  const auto script = ResolveDownloadScript(scriptName);
  if (!script.has_value()) {
    return Status::Fail(ErrorCode::InvalidArgument, "unknown download script: " + scriptName);
  }
  std::error_code errorCode;
  if (!fs::exists(*script, errorCode)) {
    return Status::Fail(ErrorCode::NotFound, "script not found: " + *script);
  }

  std::lock_guard<std::mutex> lifecycle(lifecycleMutex_);
  if (downloading_.load()) {
    return Status::Fail(ErrorCode::InvalidArgument, "download already running");
  }
  JoinFinishedWorkers();
  downloading_.store(true);
  const std::string command = config_.pythonCommand + " \"" + *script + "\" 2>&1";
  downloadWorker_ = std::thread([this, command]() { RunDownload(command); });
  return Status::Ok();
}

void TrainingService::RunDownload(std::string command) {
  AppendLog("[download] " + command);
  FILE* pipe = LLM_POPEN(command.c_str(), "r");
  if (pipe == nullptr) {
    AppendLog("[download] ERROR: could not start python");
    downloading_.store(false);
    return;
  }

  std::array<char, 1024> buffer{};
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    const std::string line = TrimCopy(buffer.data());
    if (!line.empty()) {
      AppendLog("[download] " + line);
    }
  }
  const int exitCode = LLM_PCLOSE(pipe);
  AppendLog(exitCode == 0 ? "[download] finished" : "[download] failed with exit code " + std::to_string(exitCode));
  downloading_.store(false);
}

void TrainingService::Shutdown() {
  std::lock_guard<std::mutex> lifecycle(lifecycleMutex_);
  stopRequested_.store(true);
  if (worker_.joinable()) {
    worker_.join();
  }
  if (downloadWorker_.joinable()) {
    downloadWorker_.join();
  }
}

} // namespace llm::server
