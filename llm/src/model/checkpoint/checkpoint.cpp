#include "llm/model/checkpoint/checkpoint.hpp"

#include <cstring>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace llm::model {

namespace {

struct CheckpointHeader {
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

[[nodiscard]] Status WriteTensor(std::ofstream& output, const std::string& name, const Tensor& tensor) {
  const Status nameStatus = WriteString(output, name);
  if (!nameStatus.IsOk()) {
    return nameStatus;
  }

  const auto rank = static_cast<std::uint32_t>(tensor.Rank());
  if (!WriteBytes(output, &rank, sizeof(rank))) {
    return Status::Fail(ErrorCode::IoError, "failed to write tensor rank");
  }

  for (Dimension axis = 0; axis < tensor.Rank(); ++axis) {
    const auto dim = static_cast<std::uint64_t>(tensor.GetShape()[axis]);
    if (!WriteBytes(output, &dim, sizeof(dim))) {
      return Status::Fail(ErrorCode::IoError, "failed to write tensor dimension");
    }
  }

  const auto numel = static_cast<std::uint64_t>(tensor.Numel());
  if (!WriteBytes(output, &numel, sizeof(numel))) {
    return Status::Fail(ErrorCode::IoError, "failed to write tensor numel");
  }

  if (numel > 0 && !WriteBytes(output, tensor.Data(), numel * sizeof(Scalar))) {
    return Status::Fail(ErrorCode::IoError, "failed to write tensor data");
  }

  return Status::Ok();
}

[[nodiscard]] Result<Tensor> ReadTensor(std::ifstream& input, std::string& outName) {
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

  std::vector<Scalar> data(static_cast<std::size_t>(numel));
  if (numel > 0 && !ReadBytes(input, data.data(), static_cast<std::size_t>(numel) * sizeof(Scalar))) {
    return Result<Tensor>::Fail(ErrorCode::IoError, "failed to read tensor data");
  }

  return Result<Tensor>::Ok(Tensor::FromBuffer(Shape(dims), std::move(data)));
}

[[nodiscard]] ModelConfig HeaderToConfig(const CheckpointHeader& header) {
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

[[nodiscard]] CheckpointHeader ConfigToHeader(const ModelConfig& config, const std::uint32_t numTensors) {
  CheckpointHeader header{};
  std::memcpy(header.magic, kCheckpointMagic, sizeof(header.magic));
  header.version = kCheckpointVersion;
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

void CollectExpectedTensorNames(const TransformerModel& model, std::vector<std::string>& names) {
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

[[nodiscard]] Status AssignTensor(TransformerModel& model, const std::string& name, const Tensor& tensor) {
  if (name == "token_embedding") {
    if (tensor.GetShape() != model.TokenEmbeddingMutable().GetShape()) {
      return Status::Fail(ErrorCode::InvalidArgument, "token_embedding shape mismatch");
    }
    model.TokenEmbeddingMutable() = tensor;
    return Status::Ok();
  }

  if (name == "final_norm_weight") {
    if (tensor.GetShape() != model.FinalNormWeightMutable().GetShape()) {
      return Status::Fail(ErrorCode::InvalidArgument, "final_norm_weight shape mismatch");
    }
    model.FinalNormWeightMutable() = tensor;
    return Status::Ok();
  }

  if (name == "lm_head_weight") {
    if (tensor.GetShape() != model.LmHeadWeightMutable().GetShape()) {
      return Status::Fail(ErrorCode::InvalidArgument, "lm_head_weight shape mismatch");
    }
    model.LmHeadWeightMutable() = tensor;
    return Status::Ok();
  }

  const std::string layerPrefix = "layer.";
  if (name.rfind(layerPrefix, 0) != 0) {
    return Status::Fail(ErrorCode::InvalidArgument, "unknown tensor name: " + name);
  }

  const auto dotAfterLayer = name.find('.', layerPrefix.size());
  if (dotAfterLayer == std::string::npos) {
    return Status::Fail(ErrorCode::InvalidArgument, "invalid layer tensor name: " + name);
  }

  const std::size_t layerIndex = static_cast<std::size_t>(std::stoul(name.substr(layerPrefix.size(), dotAfterLayer - layerPrefix.size())));
  if (layerIndex >= model.NumLayers()) {
    return Status::Fail(ErrorCode::OutOfRange, "layer index out of range");
  }

  TransformerBlock& block = model.Layer(layerIndex);
  const std::string suffix = name.substr(dotAfterLayer + 1);

  if (suffix == "attn_norm_weight") {
    if (tensor.GetShape() != block.AttnNormWeightMutable().GetShape()) {
      return Status::Fail(ErrorCode::InvalidArgument, "attn_norm_weight shape mismatch");
    }
    block.AttnNormWeightMutable() = tensor;
    return Status::Ok();
  }

  if (suffix == "ffn_norm_weight") {
    if (tensor.GetShape() != block.FfnNormWeightMutable().GetShape()) {
      return Status::Fail(ErrorCode::InvalidArgument, "ffn_norm_weight shape mismatch");
    }
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

[[nodiscard]] Status WriteModelTensors(std::ofstream& output, const TransformerModel& model) {
  const Status embeddingStatus = WriteTensor(output, "token_embedding", model.TokenEmbedding());
  if (!embeddingStatus.IsOk()) {
    return embeddingStatus;
  }

  const Status finalNormStatus = WriteTensor(output, "final_norm_weight", model.FinalNormWeight());
  if (!finalNormStatus.IsOk()) {
    return finalNormStatus;
  }

  const Status headStatus = WriteTensor(output, "lm_head_weight", model.LmHeadWeight());
  if (!headStatus.IsOk()) {
    return headStatus;
  }

  for (std::size_t layer = 0; layer < model.NumLayers(); ++layer) {
    const TransformerBlock& block = model.Layer(layer);
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
      const Status status = WriteTensor(output, tensorName, *tensor);
      if (!status.IsOk()) {
        return status;
      }
    }
  }

  return Status::Ok();
}

} // namespace

Status Checkpoint::Save(const TransformerModel& model, const std::string& path) {
  std::vector<std::string> expectedNames;
  CollectExpectedTensorNames(model, expectedNames);

  const CheckpointHeader header = ConfigToHeader(model.GetConfig(), static_cast<std::uint32_t>(expectedNames.size()));

  std::ofstream output(path, std::ios::binary);
  if (!output) {
    return Status::Fail(ErrorCode::IoError, "cannot open checkpoint for writing: " + path);
  }

  if (!WriteBytes(output, &header, sizeof(header))) {
    return Status::Fail(ErrorCode::IoError, "failed to write checkpoint header");
  }

  return WriteModelTensors(output, model);
}

Result<TransformerModel> Checkpoint::Load(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Result<TransformerModel>::Fail(ErrorCode::IoError, "cannot open checkpoint for reading: " + path);
  }

  CheckpointHeader header{};
  if (!ReadBytes(input, &header, sizeof(header))) {
    return Result<TransformerModel>::Fail(ErrorCode::IoError, "failed to read checkpoint header");
  }

  if (std::memcmp(header.magic, kCheckpointMagic, sizeof(header.magic)) != 0) {
    return Result<TransformerModel>::Fail(ErrorCode::InvalidArgument, "invalid checkpoint magic");
  }

  if (header.version != kCheckpointVersion) {
    return Result<TransformerModel>::Fail(ErrorCode::InvalidArgument, "unsupported checkpoint version");
  }

  const ModelConfig config = HeaderToConfig(header);
  const Status validation = config.Validate();
  if (!validation.IsOk()) {
    return Result<TransformerModel>::Fail(validation.GetError().code, validation.Message());
  }

  TransformerModel model(config);
  std::unordered_map<std::string, bool> loaded;

  for (std::uint32_t tensorIndex = 0; tensorIndex < header.numTensors; ++tensorIndex) {
    std::string name;
    auto tensor = ReadTensor(input, name);
    if (!tensor.IsOk()) {
      return Result<TransformerModel>::Fail(tensor.GetError().code, tensor.GetError().message);
    }

    const Status assignStatus = AssignTensor(model, name, tensor.Value());
    if (!assignStatus.IsOk()) {
      return Result<TransformerModel>::Fail(assignStatus.GetError().code, assignStatus.Message());
    }

    loaded[name] = true;
  }

  std::vector<std::string> expectedNames;
  CollectExpectedTensorNames(model, expectedNames);
  for (const std::string& expected : expectedNames) {
    if (!loaded.contains(expected)) {
      return Result<TransformerModel>::Fail(ErrorCode::InvalidArgument, "missing tensor in checkpoint: " + expected);
    }
  }

  return Result<TransformerModel>::Ok(std::move(model));
}

} // namespace llm::model
