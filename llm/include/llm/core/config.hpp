#pragma once

#include <cstddef>
#include <string>

#include "llm/core/status.hpp"

namespace llm {

struct ModelConfig {
  std::size_t vocabSize = 32000;
  std::size_t hiddenDim = 512;
  std::size_t numLayers = 6;
  std::size_t numHeads = 8;
  std::size_t numKvHeads = 8;
  std::size_t intermediateDim = 2048;
  std::size_t maxSeqLen = 2048;
  float ropeTheta = 10000.0f;
  float normEps = 1e-5f;

  [[nodiscard]] Status Validate() const;
};

class Config {
public:
  [[nodiscard]] static ModelConfig DefaultModelConfig();

  [[nodiscard]] static Result<ModelConfig> LoadFromFile(const std::string& path);

  [[nodiscard]] static Status SaveToFile(const ModelConfig& config, const std::string& path);
};

} // namespace llm
