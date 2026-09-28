#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "llm/training/training_session.hpp"

namespace {

namespace fs = std::filesystem;
using llm::training::ComputeSessionKey;
using llm::training::TrainingJobConfig;
using llm::training::TrainingRunner;

class TrainingSessionTest : public ::testing::Test {
protected:
  void SetUp() override {
    root_ = fs::temp_directory_path() / "maya_training_session_test";
    fs::remove_all(root_);
    fs::create_directories(root_);
    datasetA_ = WriteDataset("a.jsonl", {{"merhaba", "Merhaba, nasil yardimci olabilirim?"},
                                         {"python nedir", "Python bir programlama dilidir."},
                                         {"tesekkurler", "Rica ederim."}});
    datasetB_ = WriteDataset("b.jsonl", {{"selam", "Selam! Buradayim."}, {"git nedir", "Versiyon kontrol sistemi."}});
  }

  void TearDown() override { fs::remove_all(root_); }

  std::string WriteDataset(const std::string& name, const std::vector<std::pair<std::string, std::string>>& pairs) {
    const fs::path path = root_ / name;
    std::ofstream output(path);
    output << "# test dataset\n";
    for (const auto& [prompt, response] : pairs) {
      output << R"({"intent":"chat","messages":[{"role":"user","content":")" << prompt
             << R"("},{"role":"assistant","content":")" << response << R"("}]})" << '\n';
    }
    return path.generic_string();
  }

  TrainingJobConfig TinyConfig(const std::vector<std::string>& datasets) const {
    TrainingJobConfig config;
    config.datasetPaths = datasets;
    config.includeBuiltin = false;
    config.model.vocabSize = 96;
    config.model.hiddenDim = 16;
    config.model.numLayers = 1;
    config.model.numHeads = 2;
    config.model.numKvHeads = 1;
    config.model.intermediateDim = 32;
    config.model.maxSeqLen = 16;
    config.steps = 4;
    config.checkpointInterval = 1;
    config.workDir = (root_ / "work").generic_string();
    config.outputModelPath = (root_ / "out.ckpt").generic_string();
    config.outputQuantPath = (root_ / "out.ckptq").generic_string();
    config.outputTokenizerPath = (root_ / "tokenizer").generic_string();
    return config;
  }

  fs::path root_;
  std::string datasetA_;
  std::string datasetB_;
};

TEST_F(TrainingSessionTest, SessionKeyIsStableAndSensitiveToInputs) {
  const TrainingJobConfig base = TinyConfig({datasetA_});
  EXPECT_EQ(ComputeSessionKey(base), ComputeSessionKey(base));

  TrainingJobConfig reordered = TinyConfig({datasetB_, datasetA_});
  TrainingJobConfig ordered = TinyConfig({datasetA_, datasetB_});
  EXPECT_EQ(ComputeSessionKey(reordered), ComputeSessionKey(ordered));
  EXPECT_NE(ComputeSessionKey(base), ComputeSessionKey(ordered));

  TrainingJobConfig deeper = base;
  deeper.model.numLayers = 2;
  EXPECT_NE(ComputeSessionKey(base), ComputeSessionKey(deeper));

  TrainingJobConfig differentInterval = base;
  differentInterval.checkpointInterval = 50;
  EXPECT_EQ(ComputeSessionKey(base), ComputeSessionKey(differentInterval));
}

TEST_F(TrainingSessionTest, ConfigJsonRoundTrip) {
  TrainingJobConfig config = TinyConfig({datasetA_, datasetB_});
  config.learningRate = 0.0025f;
  config.seed = 7;

  const std::string json = "{" + llm::training::SerializeTrainingJobConfigFields(config) + "}";
  const TrainingJobConfig parsed = llm::training::ParseTrainingJobConfig(json, TrainingJobConfig{});

  EXPECT_EQ(parsed.datasetPaths, config.datasetPaths);
  EXPECT_EQ(parsed.includeBuiltin, config.includeBuiltin);
  EXPECT_EQ(parsed.model.numLayers, config.model.numLayers);
  EXPECT_EQ(parsed.model.hiddenDim, config.model.hiddenDim);
  EXPECT_EQ(parsed.model.numKvHeads, config.model.numKvHeads);
  EXPECT_EQ(parsed.model.vocabSize, config.model.vocabSize);
  EXPECT_EQ(parsed.steps, config.steps);
  EXPECT_NEAR(parsed.learningRate, config.learningRate, 1e-6f);
  EXPECT_EQ(parsed.seed, config.seed);
  EXPECT_EQ(parsed.workDir, config.workDir);
  EXPECT_EQ(parsed.outputQuantPath, config.outputQuantPath);
  EXPECT_EQ(ComputeSessionKey(parsed), ComputeSessionKey(config));
}

TEST_F(TrainingSessionTest, StopsSavesAndResumesThenPublishes) {
  const TrainingJobConfig config = TinyConfig({datasetA_});

  std::atomic<bool> stop{false};
  TrainingRunner first(config, nullptr, [&stop](const llm::training::TrainingProgress& progress) {
    if (progress.step == 2) {
      stop.store(true);
    }
  });
  ASSERT_TRUE(first.Run(stop).IsOk());
  EXPECT_TRUE(first.WasStopped());
  EXPECT_EQ(first.State().completedSteps, 2u);
  EXPECT_TRUE(fs::exists(llm::training::SessionCheckpointPath(config.workDir)));
  EXPECT_TRUE(fs::exists(llm::training::SessionOptimizerPath(config.workDir)));
  EXPECT_FALSE(fs::exists(config.outputModelPath));

  const auto pending = llm::training::LoadSessionState(config.workDir);
  ASSERT_TRUE(pending.has_value());
  EXPECT_EQ(pending->completedSteps, 2u);
  EXPECT_EQ(pending->totalSteps, 4u);
  EXPECT_EQ(pending->sessionKey, ComputeSessionKey(config));

  std::atomic<bool> noStop{false};
  std::vector<std::string> logs;
  TrainingRunner second(config, [&logs](const std::string& line) { logs.push_back(line); });
  ASSERT_TRUE(second.Run(noStop).IsOk());
  EXPECT_FALSE(second.WasStopped());
  EXPECT_EQ(second.ResumedFromStep(), 2u);
  EXPECT_TRUE(std::any_of(logs.begin(), logs.end(), [](const std::string& line) {
    return line.find("Optimizer state restored (2 steps)") != std::string::npos;
  }));
  EXPECT_EQ(second.State().completedSteps, 4u);
  EXPECT_TRUE(fs::exists(config.outputModelPath));
  EXPECT_TRUE(fs::exists(config.outputQuantPath));
  EXPECT_TRUE(fs::is_directory(config.outputTokenizerPath));
  EXPECT_FALSE(fs::exists(config.workDir));
}

TEST_F(TrainingSessionTest, DifferentDatasetClearsPendingSession) {
  const TrainingJobConfig configA = TinyConfig({datasetA_});
  std::atomic<bool> stop{false};
  TrainingRunner partial(configA, nullptr, [&stop](const llm::training::TrainingProgress& progress) {
    if (progress.step == 1) {
      stop.store(true);
    }
  });
  ASSERT_TRUE(partial.Run(stop).IsOk());
  ASSERT_TRUE(llm::training::LoadSessionState(configA.workDir).has_value());

  std::vector<std::string> logs;
  std::atomic<bool> noStop{false};
  TrainingRunner fresh(TinyConfig({datasetB_}), [&logs](const std::string& line) { logs.push_back(line); });
  ASSERT_TRUE(fresh.Run(noStop).IsOk());
  EXPECT_EQ(fresh.ResumedFromStep(), 0u);
  EXPECT_EQ(fresh.State().completedSteps, 4u);

  const bool cleared = std::any_of(logs.begin(), logs.end(), [](const std::string& line) {
    return line.find("clearing temp checkpoint") != std::string::npos;
  });
  EXPECT_TRUE(cleared);
}

} // namespace
