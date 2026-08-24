#include "llm/training/parameter.hpp"

namespace llm::training {

void ParameterList::Clear() {
  parameters_.clear();
}

void ParameterList::Add(Tensor& tensor) {
  parameters_.push_back(Parameter{&tensor, Tensor::Zeros(tensor.GetShape())});
}

void ParameterList::ZeroGrad() {
  for (Parameter& parameter : parameters_) {
    parameter.grad.Fill(0.0f);
  }
}

void ParameterList::CollectFromModel(model::TransformerModel& model, ParameterList& parameters) {
  parameters.Clear();
  parameters.Add(model.TokenEmbeddingMutable());
  parameters.Add(model.FinalNormWeightMutable());
  parameters.Add(model.LmHeadWeightMutable());

  for (std::size_t layer = 0; layer < model.NumLayers(); ++layer) {
    model::TransformerBlock& block = model.Layer(layer);
    parameters.Add(block.AttnNormWeightMutable());
    parameters.Add(block.FfnNormWeightMutable());

    parameters.Add(block.AttentionModule().QueryProjection().WeightMutable());
    parameters.Add(block.AttentionModule().KeyProjection().WeightMutable());
    parameters.Add(block.AttentionModule().ValueProjection().WeightMutable());
    parameters.Add(block.AttentionModule().OutputProjection().WeightMutable());

    parameters.Add(block.FfnModule().GateProjection().WeightMutable());
    parameters.Add(block.FfnModule().UpProjection().WeightMutable());
    parameters.Add(block.FfnModule().DownProjection().WeightMutable());
  }
}

} // namespace llm::training
