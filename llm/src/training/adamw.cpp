#include "llm/training/adamw.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace llm::training {

namespace {

constexpr char kStateMagic[8] = {'L', 'L', 'M', 'A', 'D', 'A', 'M', '1'};

struct StateHeader {
  char magic[8];
  std::uint64_t step = 0;
  std::uint64_t tensorCount = 0;
};

[[nodiscard]] bool WriteRaw(std::ofstream& out, const void* data, const std::size_t size) {
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  return out.good();
}

[[nodiscard]] bool ReadRaw(std::ifstream& in, void* data, const std::size_t size) {
  in.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
  return in.good();
}

[[nodiscard]] bool WriteTensorData(std::ofstream& out, const Tensor& tensor) {
  const auto numel = static_cast<std::uint64_t>(tensor.Numel());
  return WriteRaw(out, &numel, sizeof(numel)) &&
         (numel == 0 || WriteRaw(out, tensor.Data(), static_cast<std::size_t>(numel) * sizeof(Scalar)));
}

[[nodiscard]] bool ReadTensorData(std::ifstream& in, Tensor& tensor) {
  std::uint64_t numel = 0;
  if (!ReadRaw(in, &numel, sizeof(numel)) || numel != static_cast<std::uint64_t>(tensor.Numel())) {
    return false;
  }
  return numel == 0 || ReadRaw(in, tensor.Data(), static_cast<std::size_t>(numel) * sizeof(Scalar));
}

} // namespace

AdamW::AdamW(AdamWConfig config) : config_(config) {}

void AdamW::Reset() {
  step_ = 0;
  moment1_.clear();
  moment2_.clear();
}

bool AdamW::MomentsMatch(const ParameterList& parameters) const {
  if (moment1_.size() != parameters.Size() || moment2_.size() != parameters.Size()) {
    return false;
  }
  for (std::size_t index = 0; index < parameters.Size(); ++index) {
    if (moment1_[index].Numel() != parameters.Parameters()[index].tensor->Numel()) {
      return false;
    }
  }
  return true;
}

void AdamW::EnsureMoments(const ParameterList& parameters) {
  if (MomentsMatch(parameters)) {
    return;
  }
  moment1_.clear();
  moment2_.clear();
  for (const Parameter& parameter : parameters.Parameters()) {
    moment1_.push_back(Tensor::Zeros(parameter.tensor->GetShape()));
    moment2_.push_back(Tensor::Zeros(parameter.tensor->GetShape()));
  }
}

Status AdamW::Step(ParameterList& parameters) {
  EnsureMoments(parameters);

  ++step_;
  const float biasCorrection1 = 1.0f - std::pow(config_.beta1, static_cast<float>(step_));
  const float biasCorrection2 = 1.0f - std::pow(config_.beta2, static_cast<float>(step_));

  for (std::size_t index = 0; index < parameters.Size(); ++index) {
    Parameter& parameter = parameters.Parameters()[index];
    Tensor& moment1 = moment1_[index];
    Tensor& moment2 = moment2_[index];
    // nanoGPT / HF: weight decay only on matrices (rank >= 2), not RMSNorm/bias
    const bool applyDecay = parameter.tensor->Rank() >= 2;

    for (Index element = 0; element < static_cast<Index>(parameter.tensor->Numel()); ++element) {
      const Scalar gradient = parameter.grad[element];
      moment1[element] = config_.beta1 * moment1[element] + (1.0f - config_.beta1) * gradient;
      moment2[element] = config_.beta2 * moment2[element] + (1.0f - config_.beta2) * gradient * gradient;

      const Scalar mHat = moment1[element] / biasCorrection1;
      const Scalar vHat = moment2[element] / biasCorrection2;
      const Scalar update = config_.learningRate * mHat / (std::sqrt(vHat) + config_.epsilon);
      if (applyDecay) {
        (*parameter.tensor)[element] -= config_.learningRate * config_.weightDecay * (*parameter.tensor)[element];
      }
      (*parameter.tensor)[element] -= update;
    }
  }

  return Status::Ok();
}

Status AdamW::SaveState(const std::string& path) const {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return Status::Fail(ErrorCode::IoError, "cannot open optimizer state for writing: " + path);
  }

  StateHeader header{};
  std::memcpy(header.magic, kStateMagic, sizeof(header.magic));
  header.step = step_;
  header.tensorCount = moment1_.size();
  if (!WriteRaw(out, &header, sizeof(header))) {
    return Status::Fail(ErrorCode::IoError, "failed to write optimizer header");
  }

  for (std::size_t index = 0; index < moment1_.size(); ++index) {
    if (!WriteTensorData(out, moment1_[index]) || !WriteTensorData(out, moment2_[index])) {
      return Status::Fail(ErrorCode::IoError, "failed to write optimizer moments");
    }
  }
  return Status::Ok();
}

Status AdamW::LoadState(const std::string& path, const ParameterList& parameters) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::Fail(ErrorCode::IoError, "cannot open optimizer state: " + path);
  }

  StateHeader header{};
  if (!ReadRaw(in, &header, sizeof(header)) || std::memcmp(header.magic, kStateMagic, sizeof(header.magic)) != 0) {
    return Status::Fail(ErrorCode::InvalidArgument, "invalid optimizer state header");
  }
  if (header.tensorCount != parameters.Size()) {
    return Status::Fail(ErrorCode::InvalidArgument, "optimizer state does not match parameter count");
  }

  std::vector<Tensor> moment1;
  std::vector<Tensor> moment2;
  for (const Parameter& parameter : parameters.Parameters()) {
    Tensor first = Tensor::Zeros(parameter.tensor->GetShape());
    Tensor second = Tensor::Zeros(parameter.tensor->GetShape());
    if (!ReadTensorData(in, first) || !ReadTensorData(in, second)) {
      return Status::Fail(ErrorCode::InvalidArgument, "optimizer state does not match parameter shapes");
    }
    moment1.push_back(std::move(first));
    moment2.push_back(std::move(second));
  }

  step_ = static_cast<std::size_t>(header.step);
  moment1_ = std::move(moment1);
  moment2_ = std::move(moment2);
  return Status::Ok();
}

} // namespace llm::training
