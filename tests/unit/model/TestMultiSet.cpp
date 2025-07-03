/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/MultiSet.h"
#include "lucid/model/RectSet.h"
#include "lucid/util/exception.h"

using lucid::Matrix;
using lucid::MultiSet;
using lucid::RectSet;
using lucid::Set;
using lucid::Vector2;
using lucid::Vector3;

TEST(TestMultiSet, Contains) {
  const MultiSet set{RectSet{Vector2{-1, -1}, Vector2{1, 1}}, RectSet{Vector2{1, 1}, Vector2{2, 2}}};
  EXPECT_TRUE(set(Vector2{0, 0}));
  EXPECT_TRUE(set(Vector2{1.5, 1.5}));
  EXPECT_TRUE(set(Vector2{-1, -1}));
  EXPECT_TRUE(set(Vector2{1, 1}));
  EXPECT_TRUE(set(Vector2{1.1, 1}));
  EXPECT_TRUE(set(Vector2{2, 2}));
  EXPECT_FALSE(set(Vector2{2.1, 0}));
  EXPECT_FALSE(set(Vector2{-1.1, -1}));
  EXPECT_FALSE(set(Vector2{0, 1.1}));
  EXPECT_FALSE(set(Vector2{0, -1.1}));
}

TEST(TestMultiSet, EmptySet) {
  EXPECT_THROW(MultiSet{std::vector<std::unique_ptr<Set>>{}}, lucid::exception::LucidInvalidArgumentException);
}

TEST(TestMultiSet, DifferentSizeSets) {
  EXPECT_THROW(MultiSet(RectSet{Vector2{-1, -1}, Vector2{1, 1}}, RectSet{Vector3{-1, -1, -1}, Vector3{1, 1, 1}}),
               lucid::exception::LucidInvalidArgumentException);
}
