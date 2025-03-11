/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/math/RectSet.h"

using lucid::Index;
using lucid::Matrix;
using lucid::RectSet;
using lucid::Vector;
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

TEST(TestRectSet, VectorSample) {
  const RectSet set{Vector2{-1, -1}, Vector2{1, 1}, 0};
  for (int i = 0; i < 100; i++) {
    Vector2 x;
    set >> x;
    EXPECT_TRUE(set(x));
  }
}

TEST(TestRectSet, MatrixSample) {
  const RectSet set{Vector2{-1, -1}, Vector2{1, 1}, 0};
  for (int i = 0; i < 100; i++) {
    Matrix x{100, 2};
    set >> x;
    for (Index row = 0; row < x.rows(); row++) EXPECT_TRUE(set(x.row(row).transpose()));
  }
}
