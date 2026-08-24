#include <cmath>
#include <gtest/gtest.h>

#include "llm/tensor/ops.hpp"
#include "llm/tensor/tensor.hpp"

TEST(TensorOpsTest, MatMul2D) {
  llm::Tensor a = llm::Tensor::FromBuffer(llm::Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  llm::Tensor b = llm::Tensor::FromBuffer(llm::Shape{2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  llm::Tensor out = llm::Tensor::Zeros(llm::Shape{2, 2});

  ASSERT_TRUE(llm::tensor::MatMul(a, b, out).IsOk());
  EXPECT_FLOAT_EQ(out.At({0, 0}), 19.0f);
  EXPECT_FLOAT_EQ(out.At({0, 1}), 22.0f);
  EXPECT_FLOAT_EQ(out.At({1, 0}), 43.0f);
  EXPECT_FLOAT_EQ(out.At({1, 1}), 50.0f);
}

TEST(TensorOpsTest, AddElementwise) {
  llm::Tensor a = llm::Tensor::FromBuffer(llm::Shape{3}, {1.0f, 2.0f, 3.0f});
  llm::Tensor b = llm::Tensor::FromBuffer(llm::Shape{3}, {4.0f, 5.0f, 6.0f});
  llm::Tensor out = llm::Tensor::Zeros(llm::Shape{3});

  ASSERT_TRUE(llm::tensor::Add(a, b, out).IsOk());
  EXPECT_FLOAT_EQ(out[1], 7.0f);
}

TEST(TensorOpsTest, ReluZerosNegatives) {
  llm::Tensor tensor = llm::Tensor::FromBuffer(llm::Shape{4}, {-2.0f, -1.0f, 0.0f, 3.0f});
  llm::tensor::Relu(tensor);
  EXPECT_FLOAT_EQ(tensor[0], 0.0f);
  EXPECT_FLOAT_EQ(tensor[3], 3.0f);
}

TEST(TensorOpsTest, SoftmaxRowSumsToOne) {
  llm::Tensor tensor = llm::Tensor::FromBuffer(llm::Shape{2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f});
  ASSERT_TRUE(llm::tensor::Softmax(tensor, 1).IsOk());

  float row0Sum = 0.0f;
  float row1Sum = 0.0f;
  for (int col = 0; col < 3; ++col) {
    row0Sum += tensor.At({0, col});
    row1Sum += tensor.At({1, col});
  }

  EXPECT_NEAR(row0Sum, 1.0f, 1e-5f);
  EXPECT_NEAR(row1Sum, 1.0f, 1e-5f);
}

TEST(TensorOpsTest, GeluIsSmooth) {
  llm::Tensor tensor = llm::Tensor::FromBuffer(llm::Shape{1}, {-1.0f});
  llm::tensor::Gelu(tensor);
  EXPECT_GT(tensor[0], -1.0f);
  EXPECT_LT(tensor[0], 0.0f);
}
