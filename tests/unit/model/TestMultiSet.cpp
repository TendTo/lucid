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

// Test change_size with uniform expansion
TEST(TestMultiSet, ChangeSizeUniformExpansion) {
  MultiSet multi_set{RectSet{Vector2{-1, -1}, Vector2{1, 1}}, RectSet{Vector2{2, 2}, Vector2{4, 4}}};

  // Get the original bounds of the rectangles
  const auto& sets = multi_set.sets();
  const auto* rect1 = dynamic_cast<const RectSet*>(sets[0].get());
  const auto* rect2 = dynamic_cast<const RectSet*>(sets[1].get());

  const Vector2 rect1_center = (rect1->lower_bound() + rect1->upper_bound()) / 2.0;
  const Vector2 rect2_center = (rect2->lower_bound() + rect2->upper_bound()) / 2.0;

  // Expand both sets uniformly
  multi_set.change_size(Vector2{2.0, 2.0});

  // Check that centers are preserved
  const Vector2 new_rect1_center = (rect1->lower_bound() + rect1->upper_bound()) / 2.0;
  const Vector2 new_rect2_center = (rect2->lower_bound() + rect2->upper_bound()) / 2.0;

  EXPECT_DOUBLE_EQ(new_rect1_center(0), rect1_center(0));
  EXPECT_DOUBLE_EQ(new_rect1_center(1), rect1_center(1));
  EXPECT_DOUBLE_EQ(new_rect2_center(0), rect2_center(0));
  EXPECT_DOUBLE_EQ(new_rect2_center(1), rect2_center(1));

  // Check that sizes increased correctly (by 2.0 in each dimension)
  EXPECT_DOUBLE_EQ(rect1->upper_bound()(0) - rect1->lower_bound()(0), 4.0);
  EXPECT_DOUBLE_EQ(rect1->upper_bound()(1) - rect1->lower_bound()(1), 4.0);
  EXPECT_DOUBLE_EQ(rect2->upper_bound()(0) - rect2->lower_bound()(0), 4.0);
  EXPECT_DOUBLE_EQ(rect2->upper_bound()(1) - rect2->lower_bound()(1), 4.0);
}

// Test change_size with shrinking
TEST(TestMultiSet, ChangeSizeUniformShrink) {
  MultiSet multi_set{RectSet{Vector2{-2, -2}, Vector2{2, 2}}, RectSet{Vector2{3, 3}, Vector2{5, 5}}};

  const auto& sets = multi_set.sets();
  const auto* rect1 = dynamic_cast<const RectSet*>(sets[0].get());
  const auto* rect2 = dynamic_cast<const RectSet*>(sets[1].get());

  const Vector2 rect1_center = (rect1->lower_bound() + rect1->upper_bound()) / 2.0;
  const Vector2 rect2_center = (rect2->lower_bound() + rect2->upper_bound()) / 2.0;

  // Shrink both sets
  multi_set.change_size(Vector2{-2.0, -2.0});

  // Check that centers are preserved
  const Vector2 new_rect1_center = (rect1->lower_bound() + rect1->upper_bound()) / 2.0;
  const Vector2 new_rect2_center = (rect2->lower_bound() + rect2->upper_bound()) / 2.0;

  EXPECT_DOUBLE_EQ(new_rect1_center(0), rect1_center(0));
  EXPECT_DOUBLE_EQ(new_rect1_center(1), rect1_center(1));
  EXPECT_DOUBLE_EQ(new_rect2_center(0), rect2_center(0));
  EXPECT_DOUBLE_EQ(new_rect2_center(1), rect2_center(1));

  // Check that sizes decreased correctly
  EXPECT_DOUBLE_EQ(rect1->upper_bound()(0) - rect1->lower_bound()(0), 2.0);
  EXPECT_DOUBLE_EQ(rect1->upper_bound()(1) - rect1->lower_bound()(1), 2.0);
  EXPECT_DOUBLE_EQ(rect2->upper_bound()(0) - rect2->lower_bound()(0), 0.0);
  EXPECT_DOUBLE_EQ(rect2->upper_bound()(1) - rect2->lower_bound()(1), 0.0);
}

// Test change_size with non-uniform delta
TEST(TestMultiSet, ChangeSizeNonUniform) {
  MultiSet multi_set{RectSet{Vector2{0, 0}, Vector2{2, 4}}};

  const auto& sets = multi_set.sets();
  const auto* rect = dynamic_cast<const RectSet*>(sets[0].get());

  const Vector2 original_center = (rect->lower_bound() + rect->upper_bound()) / 2.0;

  // Expand with different amounts per dimension
  multi_set.change_size(Vector2{2.0, 4.0});

  // Check center preserved
  const Vector2 new_center = (rect->lower_bound() + rect->upper_bound()) / 2.0;
  EXPECT_DOUBLE_EQ(new_center(0), original_center(0));
  EXPECT_DOUBLE_EQ(new_center(1), original_center(1));

  // Check sizes
  EXPECT_DOUBLE_EQ(rect->upper_bound()(0) - rect->lower_bound()(0), 4.0);
  EXPECT_DOUBLE_EQ(rect->upper_bound()(1) - rect->lower_bound()(1), 8.0);
}

// Test change_size preserves containment
TEST(TestMultiSet, ChangeSizePreservesContainment) {
  MultiSet multi_set{RectSet{Vector2{-1, -1}, Vector2{1, 1}}, RectSet{Vector2{2, 2}, Vector2{3, 3}}};

  // Test points in the original sets
  EXPECT_TRUE(multi_set(Vector2{0, 0}));
  EXPECT_TRUE(multi_set(Vector2{2.5, 2.5}));
  EXPECT_FALSE(multi_set(Vector2{1.5, 1.5}));

  // Expand the sets
  multi_set.change_size(Vector2{2.0, 2.0});

  // Original points should still be contained
  EXPECT_TRUE(multi_set(Vector2{0, 0}));
  EXPECT_TRUE(multi_set(Vector2{2.5, 2.5}));

  // Gap might now be filled
  EXPECT_TRUE(multi_set(Vector2{1.5, 1.5}));
}

// Test change_size with zero delta (no-op)
TEST(TestMultiSet, ChangeSizeZero) {
  MultiSet multi_set{RectSet{Vector2{-1, -1}, Vector2{1, 1}}};

  const auto& sets = multi_set.sets();
  const auto* rect = dynamic_cast<const RectSet*>(sets[0].get());

  const Vector2 original_lb = rect->lower_bound();
  const Vector2 original_ub = rect->upper_bound();

  multi_set.change_size(Vector2{0.0, 0.0});

  EXPECT_DOUBLE_EQ(rect->lower_bound()(0), original_lb(0));
  EXPECT_DOUBLE_EQ(rect->lower_bound()(1), original_lb(1));
  EXPECT_DOUBLE_EQ(rect->upper_bound()(0), original_ub(0));
  EXPECT_DOUBLE_EQ(rect->upper_bound()(1), original_ub(1));
}

// Test change_size with three sets
TEST(TestMultiSet, ChangeSizeMultipleSets) {
  MultiSet multi_set{RectSet{Vector2{-1, -1}, Vector2{1, 1}}, RectSet{Vector2{2, 2}, Vector2{4, 4}},
                     RectSet{Vector2{5, 5}, Vector2{6, 6}}};

  const auto& sets = multi_set.sets();

  // Store original centers
  std::vector<Vector2> original_centers;
  for (const auto& set_ptr : sets) {
    const auto* rect = dynamic_cast<const RectSet*>(set_ptr.get());
    original_centers.push_back((rect->lower_bound() + rect->upper_bound()) / 2.0);
  }

  multi_set.change_size(Vector2{1.0, 1.0});

  // Check all centers preserved
  for (size_t i = 0; i < sets.size(); ++i) {
    const auto* rect = dynamic_cast<const RectSet*>(sets[i].get());
    const Vector2 new_center = (rect->lower_bound() + rect->upper_bound()) / 2.0;
    EXPECT_DOUBLE_EQ(new_center(0), original_centers[i](0));
    EXPECT_DOUBLE_EQ(new_center(1), original_centers[i](1));
  }
}
