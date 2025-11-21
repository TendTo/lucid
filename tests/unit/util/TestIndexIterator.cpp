/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/util/IndexIterator.h"
#include "lucid/util/exception.h"

using lucid::Index;
using lucid::IndexIterator;
using lucid::exception::LucidInvalidArgumentException;

TEST(IndexIterator, Constructor) {
  constexpr std::size_t size = 3;
  constexpr Index min_value = 0;
  constexpr Index max_value = 2;
  const IndexIterator<Index> it(size, min_value, max_value);
  EXPECT_EQ(it[0], min_value);
  EXPECT_EQ(it[1], min_value);
  EXPECT_EQ(it[2], min_value);
  EXPECT_TRUE(it);
}

TEST(IndexIterator, AllIterations) {
  std::vector<std::vector<Index>> expected{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                           {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
  std::vector<std::vector<Index>> result;
  for (IndexIterator<Index> it(3, 0, 2); it; ++it) {
    result.push_back({it[0], it[1], it[2]});
  }
  EXPECT_EQ(result, expected);
}

TEST(IndexIterator, SingleIndex) {
  std::vector<Index> expected{0, 1, 2};
  std::vector<Index> result;
  for (IndexIterator<Index> it(1, 0, 3); it; ++it) {
    result.push_back(it[0]);
  }
  EXPECT_EQ(result, expected);
}

TEST(IndexIterator, NegativeValues) {
  std::vector<std::vector<Index>> expected{{-1, -1}, {-1, 0}, {0, -1}, {0, 0}};
  std::vector<std::vector<Index>> result;
  for (IndexIterator<Index> it(2, -1, 1); it; ++it) {
    result.push_back({it[0], it[1]});
  }
  EXPECT_EQ(result, expected);
}

TEST(IndexIterator, EmptyRange) { EXPECT_THROW(IndexIterator<Index> it(0, 0, 2), LucidInvalidArgumentException); }

TEST(IndexIterator, MinValueGreaterThanMaxValue) {
  EXPECT_THROW(IndexIterator<Index> it(3, 2, 0), LucidInvalidArgumentException);
}

TEST(IndexIterator, ConstructorVector) {
  const std::vector<Index> min_value{0, 0, 0};
  const std::vector<Index> max_value{2, 2, 2};
  IndexIterator<std::vector<Index>> it(min_value, max_value);
  EXPECT_EQ(it[0], min_value[0]);
  EXPECT_EQ(it[1], min_value[1]);
  EXPECT_EQ(it[2], min_value[2]);
  EXPECT_TRUE(it);
}

TEST(IndexIterator, AllIterationsVector) {
  std::vector<std::vector<Index>> expected{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                           {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
  std::vector<std::vector<Index>> result;
  for (IndexIterator<std::vector<Index>> it(std ::vector<Index>{2, 2, 2}); it; ++it) {
    result.push_back({it[0], it[1], it[2]});
  }
  EXPECT_EQ(result, expected);
}

TEST(IndexIterator, SingleIndexVector) {
  std::vector<Index> expected{0, 1, 2};
  std::vector<Index> result;
  for (IndexIterator<std::vector<Index>> it(std::vector<Index>{3}); it; ++it) {
    result.push_back(it[0]);
  }
  EXPECT_EQ(result, expected);
}

TEST(IndexIterator, NegativeValuesVector) {
  std::vector<std::vector<Index>> expected{{-1, -1}, {-1, 0}, {0, -1}, {0, 0}};
  std::vector<std::vector<Index>> result;
  for (IndexIterator<std::vector<Index>> it(std::vector<Index>{-1, -1}, std::vector<Index>{1, 1}); it; ++it) {
    result.push_back({it[0], it[1]});
  }
  EXPECT_EQ(result, expected);
}

TEST(IndexIterator, DifferentMinMaxElementsVector) {
  std::vector<std::vector<Index>> expected{{1, 2, 3}, {1, 2, 4}, {1, 3, 3}, {1, 3, 4},
                                           {2, 2, 3}, {2, 2, 4}, {2, 3, 3}, {2, 3, 4}};
  std::vector<std::vector<Index>> result;
  for (IndexIterator<std::vector<Index>> it(std::vector<Index>{1, 2, 3}, std::vector<Index>{3, 4, 5}); it; ++it) {
    result.push_back({it[0], it[1], it[2]});
  }
  EXPECT_EQ(result, expected);
}

TEST(IndexIterator, DifferentSizeMinMaxElementsVector) {
  std::vector<std::vector<Index>> expected{{1, -2, 3}, {1, -1, 3}, {1, 0, 3}, {1, 1, 3},
                                           {2, -2, 3}, {2, -1, 3}, {2, 0, 3}, {2, 1, 3}};
  std::vector<std::vector<Index>> result;
  for (IndexIterator<std::vector<Index>> it(std::vector<Index>{1, -2, 3}, std::vector<Index>{3, 2, 4}); it; ++it) {
    result.push_back({it[0], it[1], it[2]});
  }
  EXPECT_EQ(result, expected);
}

TEST(IndexIterator, TestReset) {
  IndexIterator<Index> it(2, 0, 2);
  ++it;
  EXPECT_EQ(it.reset(), IndexIterator<Index>(2, 0, 2));
  for (; it; ++it) {
  }
  EXPECT_EQ(it.reset(), IndexIterator<Index>(2, 0, 2));
}

TEST(IndexIterator, TestResetIterations) {
  IndexIterator<Index> it(2, -5, 2);
  std::vector<std::vector<Index>> result_first;
  for (; it; ++it) {
    result_first.push_back({it[0], it[1]});
  }
  it.reset();
  std::vector<std::vector<Index>> result_second;
  for (; it; ++it) {
    result_second.push_back({it[0], it[1]});
  }
  EXPECT_EQ(result_first, result_second);
}

TEST(IndexIterator, TestResetVector) {
  IndexIterator it(std::vector<Index>{0, 0}, std::vector<Index>{2, 2});
  ++it;
  EXPECT_EQ(it.reset(), IndexIterator(std::vector<Index>{0, 0}, std::vector<Index>{2, 2}));
  for (; it; ++it) {
  }
  EXPECT_EQ(it.reset(), IndexIterator(std::vector<Index>{0, 0}, std::vector<Index>{2, 2}));
}

TEST(IndexIterator, TestResetVectorIterations) {
  IndexIterator it(std::vector<Index>{-10, 4}, std::vector<Index>{2, 6});
  std::vector<std::vector<Index>> result_first;
  for (; it; ++it) {
    result_first.push_back({it[0], it[1]});
  }
  it.reset();
  std::vector<std::vector<Index>> result_second;
  for (; it; ++it) {
    result_second.push_back({it[0], it[1]});
  }
  EXPECT_EQ(result_first, result_second);
}

TEST(IndexIterator, EmptyRangeVector) {
  EXPECT_THROW(IndexIterator<std::vector<Index>> it(std::vector<Index>{}, std::vector<Index>{}),
               LucidInvalidArgumentException);
}

TEST(IndexIterator, DifferentRangeSizeVector) {
  EXPECT_THROW(IndexIterator<std::vector<Index>> it(std::vector<Index>{1}, std::vector<Index>{2, 3}),
               LucidInvalidArgumentException);
}

TEST(IndexIterator, MinValueGreaterThanMaxValueVector) {
  EXPECT_THROW(IndexIterator<std::vector<Index>> it(std::vector<Index>{2, 2, 2}, std::vector<Index>{0, 0, 0}),
               LucidInvalidArgumentException);
}
