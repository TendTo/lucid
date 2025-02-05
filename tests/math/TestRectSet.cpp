/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/math/RectSet.h"

using lucid::Matrix;
using lucid::RectSet;
using lucid::Vector2;

TEST(TestRectSet, Contains) {
  const RectSet set{Vector2{-1, -1}, Vector2{1, 1}};
  EXPECT_TRUE(set(Vector2{0, 0}));
  EXPECT_TRUE(set(Vector2{-1, -1}));
  EXPECT_TRUE(set(Vector2{1, 1}));
  EXPECT_FALSE(set(Vector2{-1.1, -1}));
  EXPECT_FALSE(set(Vector2{1.1, 1}));
  EXPECT_FALSE(set(Vector2{0, 1.1}));
  EXPECT_FALSE(set(Vector2{0, -1.1}));
}
