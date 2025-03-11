/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/util/IndexIterator.h"
#include "lucid/util/exception.h"

using lucid::IndexIterator;
using lucid::exception::LucidInvalidArgumentException;

TEST(IndexIterator, Constructor) {
  const std::size_t size = 3;
  const long min_value = 0;
  const long max_value = 2;
  IndexIterator it(size, min_value, max_value);
  EXPECT_EQ(it[0], min_value);
  EXPECT_EQ(it[1], min_value);
  EXPECT_EQ(it[2], min_value);
  EXPECT_TRUE(it);
}

TEST(IndexIterator, AllIterations) {
  std::vector<std::vector<long>> expected{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                          {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
  std::vector<std::vector<long>> result;
  for (IndexIterator it(3, 0, 2); it; ++it) {
    result.push_back({it[0], it[1], it[2]});
  }
  EXPECT_EQ(result, expected);
}

TEST(IndexIterator, SingleIndex) {
  std::vector<long> expected{0, 1, 2};
  std::vector<long> result;
  for (IndexIterator it(1, 0, 3); it; ++it) {
    result.push_back(it[0]);
  }
  EXPECT_EQ(result, expected);
}

TEST(IndexIterator, NegativeValues) {
  std::vector<std::vector<long>> expected{{-1, -1}, {-1, 0}, {0, -1}, {0, 0}};
  std::vector<std::vector<long>> result;
  for (IndexIterator it(2, -1, 1); it; ++it) {
    result.push_back({it[0], it[1]});
  }
  EXPECT_EQ(result, expected);
}

TEST(IndexIterator, EmptyRange) { EXPECT_THROW(IndexIterator it(0, 0, 2), LucidInvalidArgumentException); }

TEST(IndexIterator, MinValueGreaterThanMaxValue) {
  EXPECT_THROW(IndexIterator it(3, 2, 0), LucidInvalidArgumentException);
}
