#include "llm/core/config.hpp"

#include <fstream>
#include <sstream>

namespace llm {

namespace {

[[nodiscard]] bool ParseSizeT(const std::string& text, std::size_t& out) {
  try {
    const auto value = std::stoull(text);
    out = static_cast<std::size_t>(value);
    return true;
  } catch (...) {
    return false;
  }
}

[[nodiscard]] bool ParseFloat(const std::string& text, float& out) {
  try {
    out = std::stof(text);
    return true;
  } catch (...) {
    return false;
  }
}

[[nodiscard]] Status ApplyKeyValue(ModelConfig& config, const std::string& key, const std::string& value) {
  if (key == "vocab_size") {
    return ParseSizeT(value, config.vocabSize) ? Status::Ok()
                                               : Status::Fail(ErrorCode::InvalidArgument, "invalid vocab_size");
  }
  if (key == "hidden_dim") {
    return ParseSizeT(value, config.hiddenDim) ? Status::Ok()
                                              : Status::Fail(ErrorCode::InvalidArgument, "invalid hidden_dim");
  }
  if (key == "num_layers") {
    return ParseSizeT(value, config.numLayers) ? Status::Ok()
                                               : Status::Fail(ErrorCode::InvalidArgument, "invalid num_layers");
  }
  if (key == "num_heads") {
    return ParseSizeT(value, config.numHeads) ? Status::Ok()
                                              : Status::Fail(ErrorCode::InvalidArgument, "invalid num_heads");
  }
  if (key == "num_kv_heads") {
    return ParseSizeT(value, config.numKvHeads) ? Status::Ok()
                                                : Status::Fail(ErrorCode::InvalidArgument, "invalid num_kv_heads");
  }
  if (key == "intermediate_dim") {
    return ParseSizeT(value, config.intermediateDim)
               ? Status::Ok()
               : Status::Fail(ErrorCode::InvalidArgument, "invalid intermediate_dim");
  }
  if (key == "max_seq_len") {
    return ParseSizeT(value, config.maxSeqLen) ? Status::Ok()
                                                : Status::Fail(ErrorCode::InvalidArgument, "invalid max_seq_len");
  }
  if (key == "rope_theta") {
    return ParseFloat(value, config.ropeTheta) ? Status::Ok()
                                               : Status::Fail(ErrorCode::InvalidArgument, "invalid rope_theta");
  }
  if (key == "norm_eps") {
    return ParseFloat(value, config.normEps) ? Status::Ok() : Status::Fail(ErrorCode::InvalidArgument, "invalid norm_eps");
  }

  return Status::Fail(ErrorCode::InvalidArgument, "unknown config key: " + key);
}

} // namespace

Status ModelConfig::Validate() const {
  if (vocabSize == 0) {
    return Status::Fail(ErrorCode::InvalidArgument, "vocab_size must be > 0");
  }
  if (hiddenDim == 0 || numLayers == 0 || numHeads == 0 || numKvHeads == 0) {
    return Status::Fail(ErrorCode::InvalidArgument, "model dimensions must be > 0");
  }
  if (hiddenDim % numHeads != 0) {
    return Status::Fail(ErrorCode::InvalidArgument, "hidden_dim must be divisible by num_heads");
  }
  if (numHeads % numKvHeads != 0) {
    return Status::Fail(ErrorCode::InvalidArgument, "num_heads must be divisible by num_kv_heads");
  }
  if (maxSeqLen == 0) {
    return Status::Fail(ErrorCode::InvalidArgument, "max_seq_len must be > 0");
  }
  if (ropeTheta <= 0.0f || normEps <= 0.0f) {
    return Status::Fail(ErrorCode::InvalidArgument, "rope_theta and norm_eps must be > 0");
  }
  return Status::Ok();
}

ModelConfig Config::DefaultModelConfig() { return ModelConfig{}; }

Result<ModelConfig> Config::LoadFromFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    return Result<ModelConfig>::Fail(ErrorCode::IoError, "cannot open config file: " + path);
  }

  ModelConfig config = DefaultModelConfig();
  std::string line;

  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    const auto delimiter = line.find('=');
    if (delimiter == std::string::npos) {
      return Result<ModelConfig>::Fail(ErrorCode::InvalidArgument, "invalid config line: " + line);
    }

    const std::string key = line.substr(0, delimiter);
    const std::string value = line.substr(delimiter + 1);

    const Status status = ApplyKeyValue(config, key, value);
    if (!status.IsOk()) {
      return Result<ModelConfig>::Fail(status.GetError().code, status.Message());
    }
  }

  const Status validation = config.Validate();
  if (!validation.IsOk()) {
    return Result<ModelConfig>::Fail(validation.GetError().code, validation.Message());
  }

  return Result<ModelConfig>::Ok(config);
}

Status Config::SaveToFile(const ModelConfig& config, const std::string& path) {
  std::ofstream output(path);
  if (!output) {
    return Status::Fail(ErrorCode::IoError, "cannot write config file: " + path);
  }

  output << "vocab_size=" << config.vocabSize << '\n'
         << "hidden_dim=" << config.hiddenDim << '\n'
         << "num_layers=" << config.numLayers << '\n'
         << "num_heads=" << config.numHeads << '\n'
         << "num_kv_heads=" << config.numKvHeads << '\n'
         << "intermediate_dim=" << config.intermediateDim << '\n'
         << "max_seq_len=" << config.maxSeqLen << '\n'
         << "rope_theta=" << config.ropeTheta << '\n'
         << "norm_eps=" << config.normEps << '\n';

  return Status::Ok();
}

} // namespace llm
