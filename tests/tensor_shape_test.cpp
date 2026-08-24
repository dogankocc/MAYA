#include <gtest/gtest.h>

#include "llm/tensor/shape.hpp"

TEST(ShapeTest, ComputesNumel) {
  const llm::Shape shape{2, 3, 4};
  EXPECT_EQ(shape.Rank(), 3U);
  EXPECT_EQ(shape.Numel(), 24U);
}

TEST(ShapeTest, ComputesRowMajorStrides) {
  const llm::Shape shape{2, 3, 4};
  ASSERT_EQ(shape.Strides().size(), 3U);
  EXPECT_EQ(shape.Strides()[0], 12U);
  EXPECT_EQ(shape.Strides()[1], 4U);
  EXPECT_EQ(shape.Strides()[2], 1U);
}

TEST(ShapeTest, ToStringFormatsDimensions) {
  const llm::Shape shape{2, 3};
  EXPECT_EQ(shape.ToString(), "(2, 3)");
}
