#pragma once

#include <cstddef>
#include <string>

#include "llm/core/status.hpp"

namespace llm::training {

struct FineTuneOptions {
  std::string corpusPath;
  std::string fp32OutputPath;
  std::string quantOutputPath;
  std::string tokenizerPath;
  std::string inputCheckpointPath;
  std::size_t stepsPerSample = 25;
  std::size_t maxSteps = 400;
  float learningRate = 0.0008f;
};

[[nodiscard]] Status RunFineTuneOnCorpus(const FineTuneOptions& options);

} // namespace llm::training
