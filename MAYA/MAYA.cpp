#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "launcher.hpp"
#include "llm/cli/chat_cli.hpp"
#include "llm/cli/chat_session.hpp"
#include "llm/llm.hpp"
#include "llm/server/auto_learn_service.hpp"
#include "llm/server/hybrid_chat_backend.hpp"
#include "llm/server/training_http_server.hpp"
#include "llm/server/training_service.hpp"
#include "llm/storage/storage_backend.hpp"
#include "llm/training/training_session.hpp"

namespace {

namespace fs = std::filesystem;

constexpr int kChatPort = 8765;
constexpr int kTrainPort = 8766;
constexpr const char* kLocalHost = "127.0.0.1";
constexpr const char* kChatAppRelativePath = "chat_app/build/windows/x64/runner/Release/llm_chat.exe";

bool HasFlag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == key) {
      return true;
    }
  }
  return false;
}

std::string GetArg(int argc, char** argv, const std::string& key, const std::string& defaultValue) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == key) {
      return argv[i + 1];
    }
  }
  return defaultValue;
}

std::size_t GetSizeArg(int argc, char** argv, const std::string& key, const std::size_t defaultValue) {
  const std::string text = GetArg(argc, argv, key, "");
  if (text.empty() || text == "auto") {
    return defaultValue;
  }
  return static_cast<std::size_t>(std::stoul(text));
}

std::string LocalUrl(const int port) {
  return std::string("http://") + kLocalHost + ":" + std::to_string(port);
}

void PrintUsage() {
  std::cout << "Usage:\n"
            << "  MAYA                (launch: chat API + chat app + trainer UI)\n"
            << "  MAYA launch [--backend openai|local] [--port N]\n"
            << "  MAYA serve [--backend openai|local] [--port N] [--host H]\n"
            << "                      [--api-host H] [--api-key K] [--model M]\n"
            << "                      [--model-path P] [--tokenizer T]\n"
            << "                      [--openai-model M]\n"
            << "                      [--auto-learn] [--no-auto-learn]\n"
            << "                      [--auto-learn-threshold N]\n"
            << "  MAYA train-server [--port N] [--host H] [--chat-url URL] [--python CMD]\n"
            << "  MAYA train [--output <model.ckpt>] [--tokenizer <dir>]\n"
            << "                      [--corpus-dir <dir>] [--corpus <file>] [--steps N|auto]\n"
            << "                      [--vocab N] [--corpus-only] [--checkpoint-every N]\n"
            << "  MAYA demo\n"
            << "  MAYA chat --model <path> --tokenizer <dir>\n"
            << "  MAYA agent --model <path> --tokenizer <dir>\n"
            << "  MAYA quantize --input <fp32.ckpt> --output <int8.ckptq>\n"
            << "  MAYA upload --file <path> --drive-client-id ... --drive-client-secret ... --drive-refresh-token ...\n"
            << "  MAYA version\n";
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

int RunUpload(int argc, char** argv) {
  const std::string filePath = GetArg(argc, argv, "--file", "");
  const std::string modelName = GetArg(argc, argv, "--name", "maya-model");
  const std::string version = GetArg(argc, argv, "--version", "latest");
  const std::string clientId = GetArg(argc, argv, "--drive-client-id", "");
  const std::string clientSecret = GetArg(argc, argv, "--drive-client-secret", "");
  const std::string refreshToken = GetArg(argc, argv, "--drive-refresh-token", "");
  const std::string folderId = GetArg(argc, argv, "--drive-folder-id", "");

  if (filePath.empty()) {
    std::cerr << "Error: --file is required\n";
    return 1;
  }
  if (clientId.empty() || clientSecret.empty() || refreshToken.empty()) {
    std::cerr << "Error: --drive-client-id, --drive-client-secret, and --drive-refresh-token are required\n";
    return 1;
  }

  llm::storage::StorageConfig config;
  config.type = llm::storage::StorageConfig::Type::GoogleDrive;
  config.driveClientId = clientId;
  config.driveClientSecret = clientSecret;
  config.driveRefreshToken = refreshToken;
  config.driveRootFolderId = folderId;

  auto backend = llm::storage::CreateStorageBackend(config);
  if (!backend) {
    std::cerr << "Error: Failed to create storage backend\n";
    return 1;
  }

  std::cout << "Uploading " << filePath << " as " << modelName << ':' << version << " to Google Drive...\n";
  const llm::Status status = backend->UploadModelCheckpoint(filePath, modelName, version);
  if (!status.IsOk()) {
    std::cerr << "Error: Upload failed: " << status.Message() << "\n";
    return 1;
  }
  std::cout << "Upload successful!\n";
  return 0;
}

int RunDownload() {
  std::cout << "Download not fully implemented yet\n";
  return 1;
}

// ---------------------------------------------------------------------------
// Training (CLI): shares TrainingRunner with the trainer UI, so it resumes too.
// ---------------------------------------------------------------------------

std::vector<std::string> CollectJsonlFiles(const std::string& directory) {
  std::vector<std::string> files;
  std::error_code errorCode;
  for (const auto& entry : fs::directory_iterator(directory, errorCode)) {
    if (entry.is_regular_file(errorCode) && entry.path().extension() == ".jsonl") {
      files.push_back(entry.path().generic_string());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

int RunTrainingJob(const llm::training::TrainingJobConfig& config) {
  std::atomic<bool> stopRequested{false};
  maya::launcher::InstallStopHandler(stopRequested);

  llm::training::TrainingRunner runner(config, [](const std::string& line) { std::cout << line << std::endl; });
  const llm::Status status = runner.Run(stopRequested);
  if (!status.IsOk()) {
    std::cerr << "Training failed: " << status.Message() << '\n';
    return 1;
  }
  return runner.WasStopped() ? 2 : 0;
}

llm::training::TrainingJobConfig BuildTrainConfigFromArgs(int argc, char** argv) {
  llm::training::TrainingJobConfig config = llm::training::DefaultTrainingJobConfig();
  config.outputModelPath = GetArg(argc, argv, "--output", config.outputModelPath);
  config.outputQuantPath = config.outputModelPath.ends_with(".ckpt")
                               ? config.outputModelPath.substr(0, config.outputModelPath.size() - 5) + ".ckptq"
                               : config.outputModelPath + "q";
  config.outputTokenizerPath = GetArg(argc, argv, "--tokenizer", config.outputTokenizerPath);
  config.includeBuiltin = !HasFlag(argc, argv, "--corpus-only");
  config.steps = GetSizeArg(argc, argv, "--steps", 0);
  config.model.vocabSize = GetSizeArg(argc, argv, "--vocab", config.model.vocabSize);
  config.checkpointInterval = GetSizeArg(argc, argv, "--checkpoint-every", 1);

  const std::string corpusDir = GetArg(argc, argv, "--corpus-dir", "data/corpus");
  if (config.includeBuiltin && corpusDir != "none" && corpusDir != "skip") {
    config.datasetPaths = CollectJsonlFiles(corpusDir);
  }
  const std::string corpusFile = GetArg(argc, argv, "--corpus", "data/chat_corpus_tr.jsonl");
  if (corpusFile != "none" && fs::exists(corpusFile)) {
    config.datasetPaths.push_back(corpusFile);
  }
  return config;
}

int RunTrain(int argc, char** argv) {
  return RunTrainingJob(BuildTrainConfigFromArgs(argc, argv));
}

int RunDemo() {
  auto& logger = llm::Logger::Instance();
  logger.SetOutput(std::cout);
  logger.SetLevel(llm::LogLevel::Info);
  logger.Info("demo", std::string(llm::kProjectName) + " [" + llm::kBuildStage + "]");
  logger.Info("demo", "training chat model with built-in corpus + data/corpus (resumable)");

  const int status = RunTrainingJob(BuildTrainConfigFromArgs(0, nullptr));
  if (status == 0) {
    logger.Info("demo", "run: MAYA (launch) or MAYA serve --backend local");
  }
  return status;
}

int RunQuantize(int argc, char** argv) {
  const std::string inputPath = GetArg(argc, argv, "--input", "model.ckpt");
  const std::string outputPath = GetArg(argc, argv, "--output", "model.ckptq");
  const llm::Status status = llm::quantization::QuantCheckpoint::ConvertFile(inputPath, outputPath);
  if (!status.IsOk()) {
    std::cerr << "Quantize failed: " << status.Message() << '\n';
    return 1;
  }
  std::cout << "Quantized checkpoint written to " << outputPath << '\n';
  return 0;
}

// ---------------------------------------------------------------------------
// Trainer server (separate process; hosts the trainer web UI)
// ---------------------------------------------------------------------------

int RunTrainServer(int argc, char** argv) {
  const std::string host = GetArg(argc, argv, "--host", kLocalHost);
  const int port = std::stoi(GetArg(argc, argv, "--port", std::to_string(kTrainPort)));

  llm::server::TrainingServiceConfig serviceConfig;
  serviceConfig.chatServerUrl = GetArg(argc, argv, "--chat-url", LocalUrl(kChatPort));
  serviceConfig.pythonCommand = GetArg(argc, argv, "--python", serviceConfig.pythonCommand);

  llm::server::TrainingService service(serviceConfig);
  llm::server::TrainingHttpServer server(service);

  std::atomic<bool> stopRequested{false};
  maya::launcher::InstallStopHandler(stopRequested);
  std::thread watchdog([&]() {
    while (!stopRequested.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    server.RequestStop();
  });

  std::cout << "Trainer server listening on http://" << host << ':' << port << '\n'
            << "Endpoints: /api/v1/train/{datasets,presets,status,logs,start,stop,session/clear,download}\n";
  const llm::Status status = server.Run(host, port);
  stopRequested.store(true);
  watchdog.join();
  if (!status.IsOk()) {
    std::cerr << "Trainer server failed: " << status.Message() << '\n';
    return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Chat API server
// ---------------------------------------------------------------------------

bool LooksLikeCheckpointPath(const std::string& value) {
  return value.find(".ckpt") != std::string::npos || value.find('\\') != std::string::npos ||
         value.find('/') != std::string::npos;
}

std::string ResolveModelPath(int argc, char** argv) {
  const std::string modelPath = GetArg(argc, argv, "--model-path", "");
  if (!modelPath.empty()) {
    return modelPath;
  }
  const std::string legacyModel = GetArg(argc, argv, "--model", "model.ckptq");
  return LooksLikeCheckpointPath(legacyModel) ? legacyModel : "model.ckptq";
}

std::string ResolveOpenAiModel(int argc, char** argv) {
  const std::string openAiModel = GetArg(argc, argv, "--openai-model", "");
  if (!openAiModel.empty()) {
    return openAiModel;
  }
  const std::string legacyModel = GetArg(argc, argv, "--model", "llama3.2");
  return LooksLikeCheckpointPath(legacyModel) ? "llama3.2" : legacyModel;
}

struct ChatServerBundle {
  std::unique_ptr<llm::server::ChatHttpServer> server;
  std::string host;
  int port = kChatPort;
};

ChatServerBundle BuildChatServer(int argc, char** argv, const std::string& defaultBackend) {
  const std::string modelPath = ResolveModelPath(argc, argv);
  const std::string tokenizerPath = GetArg(argc, argv, "--tokenizer", "tokenizer_data");
  const bool autoLearnEnabled = !HasFlag(argc, argv, "--no-auto-learn");
  const std::size_t autoLearnThreshold = GetSizeArg(argc, argv, "--auto-learn-threshold", 6);

  llm::server::HybridBackendConfig config;
  config.modelPath = modelPath;
  config.tokenizerPath = tokenizerPath;
  config.apiHost = GetArg(argc, argv, "--api-host", "127.0.0.1:11434");
  config.apiKey = GetArg(argc, argv, "--api-key", "");
  config.openAiModel = ResolveOpenAiModel(argc, argv);
  config.activeBackend = GetArg(argc, argv, "--backend", defaultBackend);

  auto hybrid = std::make_unique<llm::server::HybridChatBackend>(config);
  const llm::server::ServerConfig serverConfig = hybrid->GetConfig();

  std::unique_ptr<llm::server::AutoLearnService> autoLearn;
  if (autoLearnEnabled) {
    llm::server::AutoLearnConfig autoLearnConfig;
    autoLearnConfig.quantModelPath = modelPath;
    autoLearnConfig.tokenizerPath = tokenizerPath;
    autoLearnConfig.samplesBeforeTrain = autoLearnThreshold;
    llm::server::HybridChatBackend* backendPtr = hybrid.get();
    autoLearn = std::make_unique<llm::server::AutoLearnService>(
        autoLearnConfig, [backendPtr]() { return backendPtr->ReloadLocalModel(); });
    std::cout << "Auto-learn: enabled threshold=" << autoLearnThreshold << " corpus=" << autoLearnConfig.corpusPath
              << '\n';
  }

  ChatServerBundle bundle;
  bundle.host = GetArg(argc, argv, "--host", kLocalHost);
  bundle.port = std::stoi(GetArg(argc, argv, "--port", std::to_string(kChatPort)));
  std::cout << "Hybrid server: local=" << (serverConfig.localAvailable ? "yes" : "no")
            << " openai=" << (serverConfig.openAiAvailable ? "yes" : "no") << '\n'
            << "Active backend: " << serverConfig.backend << " model=" << serverConfig.model << '\n'
            << "LLM server listening on http://" << bundle.host << ':' << bundle.port << '\n'
            << "Endpoints: GET /api/v1/health, GET/POST /api/v1/config, POST /api/v1/chat, POST /api/v1/reset, "
               "POST /api/v1/reload\n";
  bundle.server = std::make_unique<llm::server::ChatHttpServer>(std::move(hybrid), std::move(autoLearn));
  return bundle;
}

int RunServe(int argc, char** argv) {
  ChatServerBundle bundle = BuildChatServer(argc, argv, "openai");
  const llm::Status runStatus = bundle.server->Run(bundle.host, bundle.port);
  if (!runStatus.IsOk()) {
    std::cerr << "Server failed: " << runStatus.Message() << '\n';
    return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Launch: chat API (this process) + trainer server (child process) + both UIs
// ---------------------------------------------------------------------------

void EnsureTrainerServer(const fs::path& exe) {
  const std::string trainUrl = LocalUrl(kTrainPort);
  if (maya::launcher::WaitForHttp(trainUrl, "/api/v1/train/status", 300)) {
    std::cout << "Trainer server already running at " << trainUrl << '\n';
    return;
  }
  const std::string command =
      maya::launcher::Quote(exe) + " train-server --port " + std::to_string(kTrainPort);
  std::cout << (maya::launcher::SpawnDetached(command, true) ? "Trainer server started (separate process)\n"
                                                              : "WARNING: trainer server could not be started\n");
}

void OpenUserInterfaces(const fs::path& root) {
  if (maya::launcher::WaitForHttp(LocalUrl(kChatPort), "/api/v1/health", 15000)) {
    const fs::path chatApp = root / kChatAppRelativePath;
    if (fs::exists(chatApp)) {
      std::cout << (maya::launcher::OpenWithShell(chatApp) ? "Chat app opened\n" : "WARNING: chat app failed to open\n");
    } else {
      std::cout << "Chat app not built (" << chatApp.generic_string() << "). Build: cd chat_app && flutter build windows\n";
    }
  }

  const std::string trainUrl = LocalUrl(kTrainPort);
  if (maya::launcher::WaitForHttp(trainUrl, "/api/v1/train/status", 15000)) {
    std::cout << (maya::launcher::OpenUrlAsApp(trainUrl) ? "Trainer UI opened: " + trainUrl + "\n"
                                                         : "Trainer UI: open " + trainUrl + " manually\n");
  } else {
    std::cout << "WARNING: trainer server did not answer at " << trainUrl << '\n';
  }
}

int RunLaunch(int argc, char** argv) {
  const fs::path root = fs::current_path();
  std::cout << llm::kProjectName << " [" << llm::kBuildStage << "] root=" << root.generic_string() << '\n';

  EnsureTrainerServer(maya::launcher::ExecutablePath());

  const std::string defaultBackend = fs::exists("model.ckptq") ? "local" : "openai";
  ChatServerBundle bundle = BuildChatServer(argc, argv, defaultBackend);

  std::thread uiThread([root]() { OpenUserInterfaces(root); });
  const llm::Status runStatus = bundle.server->Run(bundle.host, bundle.port);
  if (!runStatus.IsOk()) {
    std::cerr << "Chat server not started (" << runStatus.Message() << "); an instance may already be running\n";
  }
  uiThread.join();
  return runStatus.IsOk() ? 0 : 1;
}

// ---------------------------------------------------------------------------
// CLI chat / agent
// ---------------------------------------------------------------------------

int RunChat(int argc, char** argv) {
  llm::cli::ChatSession session;
  const llm::Status loadStatus =
      session.Load(GetArg(argc, argv, "--model", "model.ckptq"), GetArg(argc, argv, "--tokenizer", "tokenizer_data"));
  if (!loadStatus.IsOk()) {
    std::cerr << "Load failed: " << loadStatus.Message() << '\n';
    return 1;
  }
  llm::cli::ChatCli cli(session);
  return cli.Run();
}

int RunAgent(int argc, char** argv) {
  llm::agent::LocalAgentBackend agent;
  llm::agent::AgentConfig config;
  config.modelPath = GetArg(argc, argv, "--model", "model.ckptq");
  config.tokenizerPath = GetArg(argc, argv, "--tokenizer", "tokenizer_data");
  config.enableTools = true;

  const llm::Status loadStatus = agent.Load(config);
  if (!loadStatus.IsOk()) {
    std::cerr << "Agent load failed: " << loadStatus.Message() << '\n';
    return 1;
  }
  llm::cli::AgentCli cli(agent);
  return cli.Run();
}

int RunVersion() {
  std::cout << llm::kProjectName << " [" << llm::kBuildStage << "]\n";
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  const std::string command = argc < 2 ? "launch" : argv[1];

  // All commands use project-relative paths (data/, model.ckpt, trainer_web/...).
  std::error_code errorCode;
  fs::current_path(maya::launcher::ResolveProjectRoot(), errorCode);

  if (command == "launch") {
    return RunLaunch(argc, argv);
  }
  if (command == "serve") {
    return RunServe(argc, argv);
  }
  if (command == "train-server") {
    return RunTrainServer(argc, argv);
  }
  if (command == "train") {
    return RunTrain(argc, argv);
  }
  if (command == "demo") {
    return RunDemo();
  }
  if (command == "quantize") {
    return RunQuantize(argc, argv);
  }
  if (command == "chat") {
    return RunChat(argc, argv);
  }
  if (command == "agent") {
    return RunAgent(argc, argv);
  }
  if (command == "upload") {
    return RunUpload(argc, argv);
  }
  if (command == "download") {
    return RunDownload();
  }
  if (command == "version") {
    return RunVersion();
  }

  PrintUsage();
  return 1;
}
