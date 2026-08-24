#include <gtest/gtest.h>

#include "llm/tensor/tensor.hpp"

TEST(TensorTest, ZerosCreatesFilledTensor) {
  llm::Tensor tensor = llm::Tensor::Zeros(llm::Shape{2, 2});
  EXPECT_EQ(tensor.Numel(), 4U);
  EXPECT_FLOAT_EQ(tensor[0], 0.0f);
  EXPECT_FLOAT_EQ(tensor[3], 0.0f);
}

TEST(TensorTest, AtIndexesElements) {
  llm::Tensor tensor = llm::Tensor::Zeros(llm::Shape{2, 3});
  tensor.At({0, 2}) = 5.0f;
  tensor.At({1, 0}) = 7.0f;
  EXPECT_FLOAT_EQ(tensor.At({0, 2}), 5.0f);
  EXPECT_FLOAT_EQ(tensor.At({1, 0}), 7.0f);
}

TEST(TensorTest, ReshapePreservesData) {
  llm::Tensor tensor = llm::Tensor::FromBuffer(llm::Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  ASSERT_TRUE(tensor.Reshape(llm::Shape{4}).IsOk());
  EXPECT_EQ(tensor.GetShape().Numel(), 4U);
  EXPECT_FLOAT_EQ(tensor[2], 3.0f);
}

TEST(TensorTest, Transpose2DSwapsAxes) {
  llm::Tensor tensor = llm::Tensor::FromBuffer(llm::Shape{2, 3}, {1, 2, 3, 4, 5, 6});
  const llm::Tensor transposed = tensor.Transpose2D();
  EXPECT_EQ(transposed.GetShape(), llm::Shape({3, 2}));
  EXPECT_FLOAT_EQ(transposed.At({0, 1}), 4.0f);
  EXPECT_FLOAT_EQ(transposed.At({2, 1}), 6.0f);
}
