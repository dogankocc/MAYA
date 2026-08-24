#include <gtest/gtest.h>

#include "llm/core/status.hpp"

TEST(StatusTest, OkStatusIsOk) {
  const llm::Status status = llm::Status::Ok();
  EXPECT_TRUE(status.IsOk());
}

TEST(StatusTest, FailStatusContainsMessage) {
  const llm::Status status = llm::Status::Fail(llm::ErrorCode::InvalidArgument, "bad value");
  EXPECT_FALSE(status.IsOk());
  EXPECT_EQ(status.Message(), "bad value");
}

TEST(ResultTest, OkResultHoldsValue) {
  const auto result = llm::Result<int>::Ok(42);
  ASSERT_TRUE(result.IsOk());
  EXPECT_EQ(result.Value(), 42);
}

TEST(ResultTest, FailResultHoldsError) {
  const auto result = llm::Result<int>::Fail(llm::ErrorCode::IoError, "disk");
  EXPECT_FALSE(result.IsOk());
  EXPECT_EQ(result.GetError().code, llm::ErrorCode::IoError);
}
