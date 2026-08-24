#include "llm/tensor/shape.hpp"

#include <sstream>

namespace llm {

std::vector<Dimension> Shape::ComputeStrides(const std::vector<Dimension>& dims) {
  std::vector<Dimension> strides(dims.size());
  if (dims.empty()) {
    return strides;
  }

  Dimension stride = 1;
  for (std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(dims.size()) - 1; axis >= 0; --axis) {
    strides[static_cast<std::size_t>(axis)] = stride;
    stride *= dims[static_cast<std::size_t>(axis)];
  }

  return strides;
}

void Shape::Recompute() {
  strides_ = ComputeStrides(dims_);
  numel_ = 1;
  for (const Dimension dim : dims_) {
    numel_ *= dim;
  }
}

std::string Shape::ToString() const {
  std::ostringstream stream;
  stream << '(';
  for (std::size_t i = 0; i < dims_.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << dims_[i];
  }
  stream << ')';
  return stream.str();
}

} // namespace llm
