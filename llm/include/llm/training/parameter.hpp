#pragma once

#include <vector>

#include "llm/model/transformer.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::training {

struct Parameter {
  Tensor* tensor = nullptr;
  Tensor grad;
};

class ParameterList {
public:
  void Clear();

  void Add(Tensor& tensor);

  void ZeroGrad();

  [[nodiscard]] std::size_t Size() const { return parameters_.size(); }

  [[nodiscard]] std::vector<Parameter>& Parameters() { return parameters_; }

  [[nodiscard]] const std::vector<Parameter>& Parameters() const { return parameters_; }

  static void CollectFromModel(model::TransformerModel& model, ParameterList& parameters);

private:
  std::vector<Parameter> parameters_;
};

} // namespace llm::training
