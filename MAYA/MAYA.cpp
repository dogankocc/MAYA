#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "llm/cli/chat_cli.hpp"
#include "llm/cli/chat_session.hpp"
#include "llm/llm.hpp"
#include "llm/training/chat_corpus.hpp"
#include "llm/server/hybrid_chat_backend.hpp"
#include "llm/server/auto_learn_service.hpp"

namespace {

void PrintUsage() {
  std::cout << "Usage:\n"
            << "  MAYA chat --model <path> --tokenizer <dir>\n"
            << "  MAYA serve [--backend openai|local] [--port N] [--host H]\n"
            << "                      [--api-host H] [--api-key K] [--model M]\n"
            << "                      [--model-path P] [--tokenizer T]\n"
            << "                      [--openai-model M]\n"
            << "                      [--auto-learn] [--no-auto-learn]\n"
            << "                      [--auto-learn-threshold N]\n"
            << "  MAYA version\n"
            << "  MAYA quantize --input <fp32.ckpt> --output <int8.ckptq>\n"
            << "  MAYA train [--output <model.ckpt>] [--tokenizer <dir>]\n"
            << "                      [--corpus-dir <dir>] [--corpus <file>] [--steps N|auto]\n"
            << "                      [--vocab N] [--corpus-only]\n"
            << "  MAYA demo\n";
}

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
  if (LooksLikeCheckpointPath(legacyModel)) {
    return legacyModel;
  }

  return "model.ckptq";
}

std::string ResolveOpenAiModel(int argc, char** argv) {
  const std::string openAiModel = GetArg(argc, argv, "--openai-model", "");
  if (!openAiModel.empty()) {
    return openAiModel;
  }

  const std::string legacyModel = GetArg(argc, argv, "--model", "llama3.2");
  if (LooksLikeCheckpointPath(legacyModel)) {
    return "llama3.2";
  }

  return legacyModel;
}

llm::ModelConfig BuildChatModelConfig(const std::size_t vocabSize) {
  llm::ModelConfig config = llm::Config::DefaultModelConfig();
  config.vocabSize = vocabSize;
  config.hiddenDim = 512;
  config.numLayers = 24;
  config.numHeads = 8;
  config.numKvHeads = 4;
  config.intermediateDim = 2048;
  config.maxSeqLen = 384;
  return config;
}

std::vector<llm::training::DialogueSample> LoadTrainingCorpus(const std::string& corpusDir,
                                                              const std::string& corpusFile,
                                                              const bool corpusOnly) {
  std::vector<llm::training::DialogueSample> samples;
  std::size_t builtinCount = 0;

  if (!corpusOnly) {
    samples = llm::training::DefaultDialogueCorpus();
    builtinCount = samples.size();
  }

  const bool useDirectory =
      !corpusOnly && !corpusDir.empty() && corpusDir != "none" && corpusDir != "skip";
  if (useDirectory) {
    const auto directoryLoaded = llm::training::LoadDialogueCorpusDirectory(corpusDir);
    if (directoryLoaded.IsOk()) {
      const std::size_t before = samples.size();
      samples = llm::training::MergeDialogueCorpora(std::move(samples), directoryLoaded.Value());
      std::cout << "Loaded " << (samples.size() - before) << " unique samples from " << corpusDir << '\n';
    } else {
      std::cout << "Corpus directory not used: " << corpusDir << " (" << directoryLoaded.GetError().message
                << ")\n";
    }
  }

  if (!corpusFile.empty() && corpusFile != "none") {
    const auto fileLoaded = llm::training::LoadDialogueCorpus(corpusFile);
    if (fileLoaded.IsOk()) {
      const std::size_t before = samples.size();
      samples = llm::training::MergeDialogueCorpora(std::move(samples), fileLoaded.Value());
      std::cout << "Loaded " << (samples.size() - before) << " unique samples from " << corpusFile << '\n';
    } else {
      std::cerr << "Failed to load corpus file: " << corpusFile << " (" << fileLoaded.GetError().message
                << ")\n";
    }
  }

  if (corpusOnly) {
    std::cout << "Corpus-only mode, total unique samples: " << samples.size() << '\n';
  } else {
    std::cout << "Built-in dialogues: " << builtinCount << ", total unique: " << samples.size() << '\n';
  }
  return samples;
}

std::size_t ResolveTrainingSteps(const std::size_t requestedSteps, const std::size_t sampleCount) {
  if (requestedSteps > 0) {
    return requestedSteps;
  }
  const std::size_t autoSteps = std::min<std::size_t>(30000, std::max<std::size_t>(18000, sampleCount * 2));
  std::cout << "Auto training steps selected: " << autoSteps << '\n';
  return autoSteps;
}

int RunChatTraining(const std::string& outputPath, const std::string& tokenizerPath,
                    const std::string& corpusDir, const std::string& extraCorpusFile,
                    std::size_t steps, const std::size_t vocabTarget, const bool quantizeAfter,
                    const bool corpusOnly) {
#if !defined(_WIN32)
  setvbuf(stdout, nullptr, _IOLBF, 0);
  setvbuf(stderr, nullptr, _IOLBF, 0);
#endif

  const std::vector<llm::training::DialogueSample> samples =
      LoadTrainingCorpus(corpusDir, extraCorpusFile, corpusOnly);
  if (samples.empty()) {
    std::cerr << "No training samples available\n";
    return 1;
  }

  steps = ResolveTrainingSteps(steps, samples.size());

  llm::BpeTokenizer tokenizer;
  const std::string tokenizerCorpus = llm::training::BuildTokenizerCorpus(samples);
  if (!tokenizer.Train(tokenizerCorpus, vocabTarget).IsOk()) {
    std::cerr << "Tokenizer train failed\n";
    return 1;
  }

  llm::ModelConfig config = BuildChatModelConfig(tokenizer.GetVocabulary().Size());
  if (!tokenizer.Save(tokenizerPath).IsOk()) {
    std::cerr << "Tokenizer save failed\n";
    return 1;
  }

  llm::model::TransformerModel model(config);
  std::mt19937 rng(42);
  model.ResetParameters(rng);

  llm::training::TrainerConfig trainerConfig;
  trainerConfig.optimizer.learningRate = samples.size() > 1000 ? 0.002f : 0.003f;
  llm::training::Trainer trainer(model, trainerConfig);

  std::vector<std::vector<llm::TokenId>> batches =
      llm::training::BuildTrainingBatches(tokenizer, samples, config.maxSeqLen);
  if (batches.empty()) {
    std::cerr << "No valid training batches after tokenization\n";
    return 1;
  }
  std::cout << "Training batches: " << batches.size() << ", steps: " << steps << ", vocab: " << vocabTarget
            << ", layers=" << config.numLayers << ", hidden=" << config.hiddenDim << '\n';

  float lastLoss = 0.0f;
  std::size_t sampleIndex = 0;
  for (std::size_t step = 0; step < steps; ++step) {
    if (step > 0 && step % batches.size() == 0) {
      std::shuffle(batches.begin(), batches.end(), rng);
    }

    const llm::Status status = trainer.TrainStep(batches[sampleIndex % batches.size()], lastLoss);
    sampleIndex += 1;
    if (!status.IsOk()) {
      std::cerr << "Train step failed: " << status.Message() << '\n';
      return 1;
    }

    if ((step + 1) % 100 == 0 || step + 1 == steps) {
      std::cout << "step " << (step + 1) << '/' << steps << " loss=" << lastLoss << '\n';
      std::cout.flush();
    }
  }

  if (!llm::model::Checkpoint::Save(model, outputPath).IsOk()) {
    std::cerr << "Failed to save checkpoint\n";
    return 1;
  }

  std::cout << "Training complete. Saved " << outputPath << " and " << tokenizerPath << '\n';
  //Eğitim sonrası FP32 modelini (.ckpt) INT8'e dönüştürerek (.ckptq) kaydet
  if (quantizeAfter) {
    const std::string quantPath = outputPath.size() > 5 && outputPath.ends_with(".ckpt")
                                      ? outputPath.substr(0, outputPath.size() - 5) + ".ckptq"
                                      : outputPath + "q";
    if (!llm::quantization::QuantCheckpoint::ConvertFile(outputPath, quantPath).IsOk()) {
      std::cerr << "Failed to quantize checkpoint\n";
      return 1;
    }
    std::cout << "Quantized checkpoint written to " << quantPath << '\n';
  }
  //Sadece FP32 modelini kaydet, sıkıştırma yapma
  return 0;
}

int RunDemo() {
  auto& logger = llm::Logger::Instance();
  logger.SetOutput(std::cout);
  logger.SetLevel(llm::LogLevel::Info);
  logger.Info("demo", std::string(llm::kProjectName) + " [" + llm::kBuildStage + "]");
  logger.Info("demo", "training chat model with built-in corpus (this may take a few minutes)");

  const int status = RunChatTraining("model.ckpt", "tokenizer_data", "data/corpus", "data/chat_corpus_tr.jsonl", 0,
                                     3072, true, false);
  if (status == 0) {
    logger.Info("demo", "run: MAYA serve --model model.ckptq --tokenizer tokenizer_data");
  }
  return status;
}

int RunVersion() {
  std::cout << llm::kProjectName << " [" << llm::kBuildStage << "]\n";
  return 0;
}

int RunServe(int argc, char** argv) {
  const std::string defaultBackend = GetArg(argc, argv, "--backend", "openai");
  const std::string modelPath = ResolveModelPath(argc, argv);
  const std::string tokenizerPath = GetArg(argc, argv, "--tokenizer", "tokenizer_data");
  const std::string apiHost = GetArg(argc, argv, "--api-host", "127.0.0.1:11434");
  const std::string apiKey = GetArg(argc, argv, "--api-key", "");
  const std::string upstreamModel = ResolveOpenAiModel(argc, argv);
  const std::string host = GetArg(argc, argv, "--host", "127.0.0.1");
  const std::string portText = GetArg(argc, argv, "--port", "8765");
  const int port = std::stoi(portText);
  const bool autoLearnEnabled = !HasFlag(argc, argv, "--no-auto-learn");
  const std::string thresholdText = GetArg(argc, argv, "--auto-learn-threshold", "6");
  const std::size_t autoLearnThreshold = static_cast<std::size_t>(std::stoul(thresholdText));

  llm::server::HybridBackendConfig config;
  config.modelPath = modelPath;
  config.tokenizerPath = tokenizerPath;
  config.apiHost = apiHost;
  config.apiKey = apiKey;
  config.openAiModel = upstreamModel;
  config.activeBackend = defaultBackend;

  auto hybrid = std::make_unique<llm::server::HybridChatBackend>(config);
  const llm::server::ServerConfig serverConfig = hybrid->GetConfig();

  std::unique_ptr<llm::server::AutoLearnService> autoLearn;
  if (autoLearnEnabled) {
    llm::server::AutoLearnConfig autoLearnConfig;
    autoLearnConfig.enabled = true;
    autoLearnConfig.corpusPath = "data/corpus/auto_learn.jsonl";
    autoLearnConfig.statePath = "data/corpus/auto_learn_state.json";
    autoLearnConfig.fp32ModelPath = "model.ckpt";
    autoLearnConfig.quantModelPath = modelPath;
    autoLearnConfig.tokenizerPath = tokenizerPath;
    autoLearnConfig.samplesBeforeTrain = autoLearnThreshold;

    llm::server::HybridChatBackend* backendPtr = hybrid.get();
    autoLearn = std::make_unique<llm::server::AutoLearnService>(
        autoLearnConfig, [backendPtr]() { return backendPtr->ReloadLocalModel(); });
    std::cout << "Auto-learn: enabled threshold=" << autoLearnThreshold
              << " corpus=data/corpus/auto_learn.jsonl\n";
  }

  std::cout << "Hybrid server: local=" << (serverConfig.localAvailable ? "yes" : "no")
            << " openai=" << (serverConfig.openAiAvailable ? "yes" : "no") << '\n';
  std::cout << "Active backend: " << serverConfig.backend << " model=" << serverConfig.model << '\n';
  std::cout << "LLM server listening on http://" << host << ':' << port << '\n';
  std::cout << "Endpoints: GET /api/v1/health, GET/POST /api/v1/config, POST /api/v1/chat, POST /api/v1/reset\n";

  llm::server::ChatHttpServer server(std::move(hybrid), std::move(autoLearn));
  const llm::Status runStatus = server.Run(host, port);
  if (!runStatus.IsOk()) {
    std::cerr << "Server failed: " << runStatus.Message() << '\n';
    return 1;
  }

  return 0;
}

int RunChat(int argc, char** argv) {
  const std::string modelPath = GetArg(argc, argv, "--model", "model.ckptq");
  const std::string tokenizerPath = GetArg(argc, argv, "--tokenizer", "tokenizer_data");

  llm::cli::ChatSession session;
  const llm::Status loadStatus = session.Load(modelPath, tokenizerPath);
  if (!loadStatus.IsOk()) {
    std::cerr << "Load failed: " << loadStatus.Message() << '\n';
    return 1;
  }

  llm::cli::ChatCli cli(session);
  return cli.Run();
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

int RunTrain(int argc, char** argv) {
  const std::string outputPath = GetArg(argc, argv, "--output", "model.ckpt");
  const std::string tokenizerPath = GetArg(argc, argv, "--tokenizer", "tokenizer_data");
  const std::string corpusDir = GetArg(argc, argv, "--corpus-dir", "data/corpus");
  const std::string extraCorpusFile = GetArg(argc, argv, "--corpus", "data/chat_corpus_tr.jsonl");
  const std::string stepsText = GetArg(argc, argv, "--steps", "auto");
  const std::string vocabText = GetArg(argc, argv, "--vocab", "3072");
  const std::size_t steps = stepsText == "auto" ? 0 : static_cast<std::size_t>(std::stoul(stepsText));
  const std::size_t vocabTarget = static_cast<std::size_t>(std::stoul(vocabText));
  const bool corpusOnly = HasFlag(argc, argv, "--corpus-only");

  return RunChatTraining(outputPath, tokenizerPath, corpusDir, extraCorpusFile, steps, vocabTarget, false,
                         corpusOnly);
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    return RunDemo();
  }

  const std::string command = argv[1];
  if (command == "version") {
    return RunVersion();
  }
  if (command == "serve") {
    return RunServe(argc, argv);
  }
  if (command == "chat") {
    return RunChat(argc, argv);
  }
  if (command == "quantize") {
    return RunQuantize(argc, argv);
  }
  if (command == "train") {
    return RunTrain(argc, argv);
  }
  if (command == "demo") {
    return RunDemo();
  }

  PrintUsage();
  return 1;
}
