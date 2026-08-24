#include "llm/quantization/quant_checkpoint.hpp"

#include <cstring>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "llm/model/checkpoint/checkpoint.hpp"
#include "llm/model/transformer_block.hpp"
#include "llm/quantization/quantize.hpp"

namespace llm::quantization {

namespace {

struct QuantHeader {
  char magic[8];
  std::uint32_t version = 0;
  std::uint64_t vocabSize = 0;
  std::uint64_t hiddenDim = 0;
  std::uint64_t numLayers = 0;
  std::uint64_t numHeads = 0;
  std::uint64_t numKvHeads = 0;
  std::uint64_t intermediateDim = 0;
  std::uint64_t maxSeqLen = 0;
  float ropeTheta = 0.0f;
  float normEps = 0.0f;
  std::uint32_t numTensors = 0;
};

[[nodiscard]] bool WriteBytes(std::ofstream& output, const void* data, const std::size_t size) {
  output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  return static_cast<bool>(output);
}

[[nodiscard]] bool ReadBytes(std::ifstream& input, void* data, const std::size_t size) {
  input.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
  return static_cast<bool>(input);
}

[[nodiscard]] Status WriteString(std::ofstream& output, const std::string& value) {
  const auto length = static_cast<std::uint32_t>(value.size());
  if (!WriteBytes(output, &length, sizeof(length))) {
    return Status::Fail(ErrorCode::IoError, "failed to write string length");
  }
  if (length > 0 && !WriteBytes(output, value.data(), value.size())) {
    return Status::Fail(ErrorCode::IoError, "failed to write string data");
  }
  return Status::Ok();
}

[[nodiscard]] Result<std::string> ReadString(std::ifstream& input) {
  std::uint32_t length = 0;
  if (!ReadBytes(input, &length, sizeof(length))) {
    return Result<std::string>::Fail(ErrorCode::IoError, "failed to read string length");
  }

  std::string value(length, '\0');
  if (length > 0 && !ReadBytes(input, value.data(), length)) {
    return Result<std::string>::Fail(ErrorCode::IoError, "failed to read string data");
  }

  return Result<std::string>::Ok(std::move(value));
}

[[nodiscard]] ModelConfig HeaderToConfig(const QuantHeader& header) {
  ModelConfig config;
  config.vocabSize = static_cast<std::size_t>(header.vocabSize);
  config.hiddenDim = static_cast<std::size_t>(header.hiddenDim);
  config.numLayers = static_cast<std::size_t>(header.numLayers);
  config.numHeads = static_cast<std::size_t>(header.numHeads);
  config.numKvHeads = static_cast<std::size_t>(header.numKvHeads);
  config.intermediateDim = static_cast<std::size_t>(header.intermediateDim);
  config.maxSeqLen = static_cast<std::size_t>(header.maxSeqLen);
  config.ropeTheta = header.ropeTheta;
  config.normEps = header.normEps;
  return config;
}

[[nodiscard]] QuantHeader ConfigToHeader(const ModelConfig& config, const std::uint32_t numTensors) {
  QuantHeader header{};
  std::memcpy(header.magic, kQuantCheckpointMagic, sizeof(header.magic));
  header.version = kQuantCheckpointVersion;
  header.vocabSize = config.vocabSize;
  header.hiddenDim = config.hiddenDim;
  header.numLayers = config.numLayers;
  header.numHeads = config.numHeads;
  header.numKvHeads = config.numKvHeads;
  header.intermediateDim = config.intermediateDim;
  header.maxSeqLen = config.maxSeqLen;
  header.ropeTheta = config.ropeTheta;
  header.normEps = config.normEps;
  header.numTensors = numTensors;
  return header;
}

void CollectExpectedTensorNames(const model::TransformerModel& model, std::vector<std::string>& names) {
  names.push_back("token_embedding");
  names.push_back("final_norm_weight");
  names.push_back("lm_head_weight");

  for (std::size_t layer = 0; layer < model.NumLayers(); ++layer) {
    const std::string prefix = "layer." + std::to_string(layer);
    names.push_back(prefix + ".attn_norm_weight");
    names.push_back(prefix + ".ffn_norm_weight");
    names.push_back(prefix + ".attn.q_weight");
    names.push_back(prefix + ".attn.k_weight");
    names.push_back(prefix + ".attn.v_weight");
    names.push_back(prefix + ".attn.o_weight");
    names.push_back(prefix + ".ffn.gate_weight");
    names.push_back(prefix + ".ffn.up_weight");
    names.push_back(prefix + ".ffn.down_weight");
  }
}

[[nodiscard]] bool IsQuantizedLinearWeightName(const std::string& name);

[[nodiscard]] Status AssignTensor(model::TransformerModel& model, const std::string& name, const Tensor& tensor) {
  if (name == "token_embedding") {
    model.TokenEmbeddingMutable() = tensor;
    return Status::Ok();
  }
  if (name == "final_norm_weight") {
    model.FinalNormWeightMutable() = tensor;
    return Status::Ok();
  }
  if (name == "lm_head_weight") {
    model.LmHeadWeightMutable() = tensor;
    return Status::Ok();
  }

  const std::string layerPrefix = "layer.";
  if (name.rfind(layerPrefix, 0) != 0) {
    return Status::Fail(ErrorCode::InvalidArgument, "unknown tensor name: " + name);
  }

  const auto dotAfterLayer = name.find('.', layerPrefix.size());
  const std::size_t layerIndex = static_cast<std::size_t>(std::stoul(name.substr(layerPrefix.size(), dotAfterLayer - layerPrefix.size())));
  model::TransformerBlock& block = model.Layer(layerIndex);
  const std::string suffix = name.substr(dotAfterLayer + 1);

  if (suffix == "attn_norm_weight") {
    block.AttnNormWeightMutable() = tensor;
    return Status::Ok();
  }
  if (suffix == "ffn_norm_weight") {
    block.FfnNormWeightMutable() = tensor;
    return Status::Ok();
  }
  if (suffix == "attn.q_weight") {
    return block.AttentionModule().QueryProjection().LoadWeight(tensor);
  }
  if (suffix == "attn.k_weight") {
    return block.AttentionModule().KeyProjection().LoadWeight(tensor);
  }
  if (suffix == "attn.v_weight") {
    return block.AttentionModule().ValueProjection().LoadWeight(tensor);
  }
  if (suffix == "attn.o_weight") {
    return block.AttentionModule().OutputProjection().LoadWeight(tensor);
  }
  if (suffix == "ffn.gate_weight") {
    return block.FfnModule().GateProjection().LoadWeight(tensor);
  }
  if (suffix == "ffn.up_weight") {
    return block.FfnModule().UpProjection().LoadWeight(tensor);
  }
  if (suffix == "ffn.down_weight") {
    return block.FfnModule().DownProjection().LoadWeight(tensor);
  }

  return Status::Fail(ErrorCode::InvalidArgument, "unknown layer tensor: " + suffix);
}

[[nodiscard]] bool IsQuantizedLinearWeightName(const std::string& name) {
  if (name == "lm_head_weight") {
    return true;
  }

  static constexpr const char* kLinearSuffixes[] = {".attn.q_weight", ".attn.k_weight", ".attn.v_weight",
                                                    ".attn.o_weight",   ".ffn.gate_weight", ".ffn.up_weight",
                                                    ".ffn.down_weight"};
  for (const char* suffix : kLinearSuffixes) {
    if (name.size() >= std::strlen(suffix) &&
        name.compare(name.size() - std::strlen(suffix), std::string::npos, suffix) == 0) {
      return true;
    }
  }

  return false;
}

[[nodiscard]] Status AssignQuantizedTensor(model::TransformerModel& model, const std::string& name,
                                         const QuantizedTensor& tensor) {
  if (name == "lm_head_weight") {
    return model.LoadQuantizedLmHeadWeight(tensor);
  }

  const std::string layerPrefix = "layer.";
  if (name.rfind(layerPrefix, 0) != 0) {
    return Status::Fail(ErrorCode::InvalidArgument, "tensor is not quantized linear weight: " + name);
  }

  const auto dotAfterLayer = name.find('.', layerPrefix.size());
  const std::size_t layerIndex = static_cast<std::size_t>(std::stoul(name.substr(layerPrefix.size(), dotAfterLayer - layerPrefix.size())));
  model::TransformerBlock& block = model.Layer(layerIndex);
  const std::string suffix = name.substr(dotAfterLayer + 1);

  if (suffix == "attn.q_weight") {
    return block.AttentionModule().QueryProjection().LoadQuantizedWeight(tensor);
  }
  if (suffix == "attn.k_weight") {
    return block.AttentionModule().KeyProjection().LoadQuantizedWeight(tensor);
  }
  if (suffix == "attn.v_weight") {
    return block.AttentionModule().ValueProjection().LoadQuantizedWeight(tensor);
  }
  if (suffix == "attn.o_weight") {
    return block.AttentionModule().OutputProjection().LoadQuantizedWeight(tensor);
  }
  if (suffix == "ffn.gate_weight") {
    return block.FfnModule().GateProjection().LoadQuantizedWeight(tensor);
  }
  if (suffix == "ffn.up_weight") {
    return block.FfnModule().UpProjection().LoadQuantizedWeight(tensor);
  }
  if (suffix == "ffn.down_weight") {
    return block.FfnModule().DownProjection().LoadQuantizedWeight(tensor);
  }

  return Status::Fail(ErrorCode::InvalidArgument, "unknown quantized layer tensor: " + suffix);
}

[[nodiscard]] Status WriteQuantTensor(std::ofstream& output, const std::string& name, const Tensor& tensor) {
  const Status nameStatus = WriteString(output, name);
  if (!nameStatus.IsOk()) {
    return nameStatus;
  }

  const QuantizedTensor quantized = QuantizeSymmetric(tensor);
  const auto rank = static_cast<std::uint32_t>(quantized.shape.Rank());
  if (!WriteBytes(output, &rank, sizeof(rank))) {
    return Status::Fail(ErrorCode::IoError, "failed to write tensor rank");
  }

  for (Dimension axis = 0; axis < quantized.shape.Rank(); ++axis) {
    const auto dim = static_cast<std::uint64_t>(quantized.shape[axis]);
    if (!WriteBytes(output, &dim, sizeof(dim))) {
      return Status::Fail(ErrorCode::IoError, "failed to write tensor dimension");
    }
  }

  const auto numel = static_cast<std::uint64_t>(quantized.data.size());
  if (!WriteBytes(output, &numel, sizeof(numel))) {
    return Status::Fail(ErrorCode::IoError, "failed to write tensor numel");
  }

  if (!WriteBytes(output, &quantized.scale, sizeof(quantized.scale))) {
    return Status::Fail(ErrorCode::IoError, "failed to write tensor scale");
  }

  if (numel > 0 && !WriteBytes(output, quantized.data.data(), numel)) {
    return Status::Fail(ErrorCode::IoError, "failed to write int8 tensor data");
  }

  return Status::Ok();
}

[[nodiscard]] Result<Tensor> ReadQuantTensor(std::ifstream& input, std::string& outName) {
  auto name = ReadString(input);
  if (!name.IsOk()) {
    return Result<Tensor>::Fail(name.GetError().code, name.GetError().message);
  }
  outName = std::move(name.Value());

  std::uint32_t rank = 0;
  if (!ReadBytes(input, &rank, sizeof(rank))) {
    return Result<Tensor>::Fail(ErrorCode::IoError, "failed to read tensor rank");
  }

  std::vector<Dimension> dims(rank);
  for (std::uint32_t axis = 0; axis < rank; ++axis) {
    std::uint64_t dim = 0;
    if (!ReadBytes(input, &dim, sizeof(dim))) {
      return Result<Tensor>::Fail(ErrorCode::IoError, "failed to read tensor dimension");
    }
    dims[axis] = static_cast<Dimension>(dim);
  }

  std::uint64_t numel = 0;
  if (!ReadBytes(input, &numel, sizeof(numel))) {
    return Result<Tensor>::Fail(ErrorCode::IoError, "failed to read tensor numel");
  }

  float scale = 1.0f;
  if (!ReadBytes(input, &scale, sizeof(scale))) {
    return Result<Tensor>::Fail(ErrorCode::IoError, "failed to read tensor scale");
  }

  QuantizedTensor quantized;
  quantized.shape = Shape(dims);
  quantized.scale = scale;
  quantized.data.resize(static_cast<std::size_t>(numel));

  if (numel > 0 && !ReadBytes(input, quantized.data.data(), static_cast<std::size_t>(numel))) {
    return Result<Tensor>::Fail(ErrorCode::IoError, "failed to read int8 tensor data");
  }

  return Result<Tensor>::Ok(Dequantize(quantized));
}

[[nodiscard]] Result<QuantizedTensor> ReadQuantTensorRaw(std::ifstream& input, std::string& outName) {
  auto name = ReadString(input);
  if (!name.IsOk()) {
    return Result<QuantizedTensor>::Fail(name.GetError().code, name.GetError().message);
  }
  outName = std::move(name.Value());

  std::uint32_t rank = 0;
  if (!ReadBytes(input, &rank, sizeof(rank))) {
    return Result<QuantizedTensor>::Fail(ErrorCode::IoError, "failed to read tensor rank");
  }

  std::vector<Dimension> dims(rank);
  for (std::uint32_t axis = 0; axis < rank; ++axis) {
    std::uint64_t dim = 0;
    if (!ReadBytes(input, &dim, sizeof(dim))) {
      return Result<QuantizedTensor>::Fail(ErrorCode::IoError, "failed to read tensor dimension");
    }
    dims[axis] = static_cast<Dimension>(dim);
  }

  std::uint64_t numel = 0;
  if (!ReadBytes(input, &numel, sizeof(numel))) {
    return Result<QuantizedTensor>::Fail(ErrorCode::IoError, "failed to read tensor numel");
  }

  QuantizedTensor quantized;
  quantized.shape = Shape(dims);
  if (!ReadBytes(input, &quantized.scale, sizeof(quantized.scale))) {
    return Result<QuantizedTensor>::Fail(ErrorCode::IoError, "failed to read tensor scale");
  }

  quantized.data.resize(static_cast<std::size_t>(numel));
  if (numel > 0 && !ReadBytes(input, quantized.data.data(), static_cast<std::size_t>(numel))) {
    return Result<QuantizedTensor>::Fail(ErrorCode::IoError, "failed to read int8 tensor data");
  }

  return Result<QuantizedTensor>::Ok(std::move(quantized));
}

[[nodiscard]] Status WriteModelTensors(std::ofstream& output, const model::TransformerModel& model) {
  const std::vector<std::pair<std::string, const Tensor*>> tensors = {
      {"token_embedding", &model.TokenEmbedding()},
      {"final_norm_weight", &model.FinalNormWeight()},
      {"lm_head_weight", &model.LmHeadWeight()},
  };

  for (const auto& [name, tensor] : tensors) {
    const Status status = WriteQuantTensor(output, name, *tensor);
    if (!status.IsOk()) {
      return status;
    }
  }

  for (std::size_t layer = 0; layer < model.NumLayers(); ++layer) {
    const model::TransformerBlock& block = model.Layer(layer);
    const std::string prefix = "layer." + std::to_string(layer);
    const std::vector<std::pair<std::string, const Tensor*>> layerTensors = {
        {prefix + ".attn_norm_weight", &block.AttnNormWeight()},
        {prefix + ".ffn_norm_weight", &block.FfnNormWeight()},
        {prefix + ".attn.q_weight", &block.AttentionModule().QueryProjection().Weight()},
        {prefix + ".attn.k_weight", &block.AttentionModule().KeyProjection().Weight()},
        {prefix + ".attn.v_weight", &block.AttentionModule().ValueProjection().Weight()},
        {prefix + ".attn.o_weight", &block.AttentionModule().OutputProjection().Weight()},
        {prefix + ".ffn.gate_weight", &block.FfnModule().GateProjection().Weight()},
        {prefix + ".ffn.up_weight", &block.FfnModule().UpProjection().Weight()},
        {prefix + ".ffn.down_weight", &block.FfnModule().DownProjection().Weight()},
    };

    for (const auto& [tensorName, tensor] : layerTensors) {
      const Status status = WriteQuantTensor(output, tensorName, *tensor);
      if (!status.IsOk()) {
        return status;
      }
    }
  }

  return Status::Ok();
}

} // namespace

Status QuantCheckpoint::Save(const model::TransformerModel& model, const std::string& path) {
  std::vector<std::string> expectedNames;
  CollectExpectedTensorNames(model, expectedNames);

  const QuantHeader header = ConfigToHeader(model.GetConfig(), static_cast<std::uint32_t>(expectedNames.size()));

  std::ofstream output(path, std::ios::binary);
  if (!output) {
    return Status::Fail(ErrorCode::IoError, "cannot open quant checkpoint for writing: " + path);
  }

  if (!WriteBytes(output, &header, sizeof(header))) {
    return Status::Fail(ErrorCode::IoError, "failed to write quant checkpoint header");
  }

  return WriteModelTensors(output, model);
}

Result<model::TransformerModel> QuantCheckpoint::Load(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Result<model::TransformerModel>::Fail(ErrorCode::IoError, "cannot open quant checkpoint for reading: " + path);
  }

  QuantHeader header{};
  if (!ReadBytes(input, &header, sizeof(header))) {
    return Result<model::TransformerModel>::Fail(ErrorCode::IoError, "failed to read quant checkpoint header");
  }

  if (std::memcmp(header.magic, kQuantCheckpointMagic, sizeof(header.magic)) != 0) {
    return Result<model::TransformerModel>::Fail(ErrorCode::InvalidArgument, "invalid quant checkpoint magic");
  }

  if (header.version != kQuantCheckpointVersion) {
    return Result<model::TransformerModel>::Fail(ErrorCode::InvalidArgument, "unsupported quant checkpoint version");
  }

  const ModelConfig config = HeaderToConfig(header);
  const Status validation = config.Validate();
  if (!validation.IsOk()) {
    return Result<model::TransformerModel>::Fail(validation.GetError().code, validation.Message());
  }

  model::TransformerModel model(config);
  std::unordered_map<std::string, bool> loaded;

  for (std::uint32_t tensorIndex = 0; tensorIndex < header.numTensors; ++tensorIndex) {
    std::string name;
    auto tensor = ReadQuantTensor(input, name);
    if (!tensor.IsOk()) {
      return Result<model::TransformerModel>::Fail(tensor.GetError().code, tensor.GetError().message);
    }

    const Status assignStatus = AssignTensor(model, name, tensor.Value());
    if (!assignStatus.IsOk()) {
      return Result<model::TransformerModel>::Fail(assignStatus.GetError().code, assignStatus.Message());
    }

    loaded[name] = true;
  }

  std::vector<std::string> expectedNames;
  CollectExpectedTensorNames(model, expectedNames);
  for (const std::string& expected : expectedNames) {
    if (!loaded.contains(expected)) {
      return Result<model::TransformerModel>::Fail(ErrorCode::InvalidArgument, "missing tensor in quant checkpoint: " + expected);
    }
  }

  return Result<model::TransformerModel>::Ok(std::move(model));
}

Result<model::TransformerModel> QuantCheckpoint::LoadRuntime(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Result<model::TransformerModel>::Fail(ErrorCode::IoError, "cannot open quant checkpoint for reading: " + path);
  }

  QuantHeader header{};
  if (!ReadBytes(input, &header, sizeof(header))) {
    return Result<model::TransformerModel>::Fail(ErrorCode::IoError, "failed to read quant checkpoint header");
  }

  if (std::memcmp(header.magic, kQuantCheckpointMagic, sizeof(header.magic)) != 0) {
    return Result<model::TransformerModel>::Fail(ErrorCode::InvalidArgument, "invalid quant checkpoint magic");
  }

  if (header.version != kQuantCheckpointVersion) {
    return Result<model::TransformerModel>::Fail(ErrorCode::InvalidArgument, "unsupported quant checkpoint version");
  }

  const ModelConfig config = HeaderToConfig(header);
  const Status validation = config.Validate();
  if (!validation.IsOk()) {
    return Result<model::TransformerModel>::Fail(validation.GetError().code, validation.Message());
  }

  model::TransformerModel model(config);
  std::unordered_map<std::string, bool> loaded;

  for (std::uint32_t tensorIndex = 0; tensorIndex < header.numTensors; ++tensorIndex) {
    std::string name;
    auto quantized = ReadQuantTensorRaw(input, name);
    if (!quantized.IsOk()) {
      return Result<model::TransformerModel>::Fail(quantized.GetError().code, quantized.GetError().message);
    }

    if (IsQuantizedLinearWeightName(name)) {
      const Status assignStatus = AssignQuantizedTensor(model, name, quantized.Value());
      if (!assignStatus.IsOk()) {
        return Result<model::TransformerModel>::Fail(assignStatus.GetError().code, assignStatus.Message());
      }
    } else {
      const Status assignStatus = AssignTensor(model, name, Dequantize(quantized.Value()));
      if (!assignStatus.IsOk()) {
        return Result<model::TransformerModel>::Fail(assignStatus.GetError().code, assignStatus.Message());
      }
    }

    loaded[name] = true;
  }

  std::vector<std::string> expectedNames;
  CollectExpectedTensorNames(model, expectedNames);
  for (const std::string& expected : expectedNames) {
    if (!loaded.contains(expected)) {
      return Result<model::TransformerModel>::Fail(ErrorCode::InvalidArgument,
                                                   "missing tensor in quant checkpoint: " + expected);
    }
  }

  model.SetQuantizedRuntimeEnabled(true);
  return Result<model::TransformerModel>::Ok(std::move(model));
}

Status QuantCheckpoint::ConvertFile(const std::string& fp32Path, const std::string& quantPath) {
  const auto model = model::Checkpoint::Load(fp32Path);
  if (!model.IsOk()) {
    return Status::Fail(model.GetError().code, model.GetError().message);
  }

  return Save(model.Value(), quantPath);
}

} // namespace llm::quantization
