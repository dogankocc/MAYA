#include "llm/tensor/tensor.hpp"

#include <algorithm>
#include <stdexcept>

namespace llm {

Tensor::Tensor(Shape shape, std::vector<Scalar> data) : shape_(std::move(shape)), data_(std::move(data)) {}

Tensor Tensor::Zeros(Shape shape) {
  return Tensor(std::move(shape), std::vector<Scalar>(shape.Numel(), 0.0f));
}

Tensor Tensor::Ones(Shape shape) {
  return Tensor(std::move(shape), std::vector<Scalar>(shape.Numel(), 1.0f));
}

Tensor Tensor::FromBuffer(Shape shape, std::vector<Scalar> data) {
  if (data.size() != shape.Numel()) {
    throw std::invalid_argument("tensor data size does not match shape");
  }
  return Tensor(std::move(shape), std::move(data));
}

Index Tensor::ResolveOffset(std::initializer_list<Index> indices) const {
  if (indices.size() != shape_.Rank()) {
    throw std::out_of_range("tensor index rank mismatch");
  }

  Index offset = 0;
  std::size_t axis = 0;
  for (const Index index : indices) {
    if (index < 0 || static_cast<Dimension>(index) >= shape_[axis]) {
      throw std::out_of_range("tensor index out of range");
    }
    offset += index * static_cast<Index>(shape_.Strides()[axis]);
    ++axis;
  }

  return offset;
}

Scalar& Tensor::At(std::initializer_list<Index> indices) {
  return data_.at(static_cast<std::size_t>(ResolveOffset(indices)));
}

const Scalar& Tensor::At(std::initializer_list<Index> indices) const {
  return data_.at(static_cast<std::size_t>(ResolveOffset(indices)));
}

Status Tensor::Reshape(Shape newShape) {
  if (newShape.Numel() != shape_.Numel()) {
    return Status::Fail(ErrorCode::InvalidArgument, "reshape requires same number of elements");
  }

  shape_ = std::move(newShape);
  return Status::Ok();
}

Tensor Tensor::Transpose2D() const {
  if (shape_.Rank() != 2) {
    throw std::invalid_argument("transpose2d requires rank-2 tensor");
  }

  const Dimension rows = shape_[0];
  const Dimension cols = shape_[1];
  Tensor transposed(Shape{cols, rows}, std::vector<Scalar>(shape_.Numel()));

  for (Dimension row = 0; row < rows; ++row) {
    for (Dimension col = 0; col < cols; ++col) {
      transposed.At({static_cast<Index>(col), static_cast<Index>(row)}) = At({static_cast<Index>(row), static_cast<Index>(col)});
    }
  }

  return transposed;
}

void Tensor::Fill(Scalar value) {
  std::fill(data_.begin(), data_.end(), value);
}

} // namespace llm
