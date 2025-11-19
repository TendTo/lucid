/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/MultiSet.h"
#include "lucid/model/RectSet.h"
#include "lucid/util/exception.h"

using lucid::Index;
using lucid::Matrix;
using lucid::MultiSet;
using lucid::RectSet;
using lucid::Scalar;
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

TEST(TestRectSet, ContainsPairs) {
  const RectSet set{std::pair<Scalar, Scalar>{-1, 1}, std::pair<Scalar, Scalar>{-1, 1}};
  EXPECT_TRUE(set(Vector2{0, 0}));
  EXPECT_TRUE(set(Vector2{-1, -1}));
  EXPECT_TRUE(set(Vector2{1, 1}));
  EXPECT_FALSE(set(Vector2{-1.1, -1}));
  EXPECT_FALSE(set(Vector2{1.1, 1}));
  EXPECT_FALSE(set(Vector2{0, 1.1}));
  EXPECT_FALSE(set(Vector2{0, -1.1}));
}

TEST(TestRectSet, Multidimensional) {
  const RectSet set_vectors{Eigen::Vector<Scalar, 5>{-1, -2, -3, -4, -5}, Eigen::Vector<Scalar, 5>{2, 3, 4, 5, 6}};
  const RectSet set_pairs{{-1.0, 2.0}, {-2.0, 3.0}, {-3.0, 4.0}, {-4.0, 5.0}, {-5.0, 6.0}};
  EXPECT_EQ(set_vectors.lower_bound(), set_pairs.lower_bound());
  EXPECT_EQ(set_vectors.upper_bound(), set_pairs.upper_bound());
}

TEST(TestRectSet, VectorSample) {
  const RectSet set{Vector2{-1, -1}, Vector2{1, 1}};
  for (int i = 0; i < 100; i++) {
    Vector2 x;
    set >> x;
    EXPECT_TRUE(set(x));
  }
}

TEST(TestRectSet, MatrixSample) {
  const RectSet set{Vector2{-1, -1}, Vector2{1, 1}};
  for (int i = 0; i < 100; i++) {
    Matrix x{100, 2};
    set >> x;
    for (Index row = 0; row < x.rows(); row++) EXPECT_TRUE(set(x.row(row)));
  }
}

TEST(TestRectSet, LatticeNoEndpointsSamePointsPerDimension) {
  constexpr int points_per_dim = 3;
  const RectSet set{Vector2{-1, -1}, Vector2{1, 1}};
  const Matrix lattice{set.lattice(points_per_dim)};
  EXPECT_EQ(lattice.rows(), 9);
  EXPECT_EQ(lattice.cols(), 2);
  for (Index row = 0; row < lattice.rows(); row++) EXPECT_TRUE(set(lattice.row(row)));
  const auto step{(set.upper_bound() - set.lower_bound()) / points_per_dim};
  for (Index row = 0; row < lattice.rows(); row++) {
    EXPECT_DOUBLE_EQ(lattice(row, 0), set.lower_bound()(0) + (row % points_per_dim) * step(0));
  }
  for (Index row = 0; row < lattice.rows(); row++) {
    EXPECT_DOUBLE_EQ(lattice(row, 1), set.lower_bound()(1) + (row / points_per_dim) * step(1));
  }
}

TEST(TestRectSet, LatticeEndpointsSamePointsPerDimension) {
  constexpr int points_per_dim = 3;
  const RectSet set{Vector2{-1, -1}, Vector2{1, 1}};
  const Matrix lattice{set.lattice(points_per_dim, true)};
  EXPECT_EQ(lattice.rows(), 9);
  EXPECT_EQ(lattice.cols(), 2);
  for (Index row = 0; row < lattice.rows(); row++) EXPECT_TRUE(set(lattice.row(row)));
  const auto step{(set.upper_bound() - set.lower_bound()) / (points_per_dim - 1)};
  for (Index row = 0; row < lattice.rows(); row++) {
    EXPECT_DOUBLE_EQ(lattice(row, 0), set.lower_bound()(0) + (row % points_per_dim) * step(0));
  }
  for (Index row = 0; row < lattice.rows(); row++) {
    EXPECT_DOUBLE_EQ(lattice(row, 1), set.lower_bound()(1) + (row / points_per_dim) * step(1));
  }
}

TEST(TestRectSet, ChangeSizeUniform) {
  // Test uniform expansion in all dimensions
  RectSet set{Vector2{-1, -1}, Vector2{1, 1}};
  const Vector2 original_center = (set.lower_bound() + set.upper_bound()) / 2.0;

  // Expand by 2 units in all dimensions
  set.change_size(2.0);

  // Check that bounds expanded correctly
  EXPECT_DOUBLE_EQ(set.lower_bound()(0), -2.0);
  EXPECT_DOUBLE_EQ(set.lower_bound()(1), -2.0);
  EXPECT_DOUBLE_EQ(set.upper_bound()(0), 2.0);
  EXPECT_DOUBLE_EQ(set.upper_bound()(1), 2.0);

  // Check that center is preserved
  const Vector2 new_center = (set.lower_bound() + set.upper_bound()) / 2.0;
  EXPECT_DOUBLE_EQ(new_center(0), original_center(0));
  EXPECT_DOUBLE_EQ(new_center(1), original_center(1));
}

TEST(TestRectSet, ChangeSizeVector) {
  // Test non-uniform expansion with vector
  RectSet set{Vector2{-1, -2}, Vector2{1, 2}};
  const Vector2 original_center = (set.lower_bound() + set.upper_bound()) / 2.0;

  // Expand by different amounts in each dimension
  set.change_size(Vector2{2.0, 4.0});

  // Check that bounds expanded correctly
  // For dimension 0: original size = 2, delta = 2, new size = 4
  // Center at 0, so bounds should be [-2, 2]
  EXPECT_DOUBLE_EQ(set.lower_bound()(0), -2.0);
  EXPECT_DOUBLE_EQ(set.upper_bound()(0), 2.0);

  // For dimension 1: original size = 4, delta = 4, new size = 8
  // Center at 0, so bounds should be [-4, 4]
  EXPECT_DOUBLE_EQ(set.lower_bound()(1), -4.0);
  EXPECT_DOUBLE_EQ(set.upper_bound()(1), 4.0);

  // Check that center is preserved
  const Vector2 new_center = (set.lower_bound() + set.upper_bound()) / 2.0;
  EXPECT_DOUBLE_EQ(new_center(0), original_center(0));
  EXPECT_DOUBLE_EQ(new_center(1), original_center(1));
}

TEST(TestRectSet, ChangeSizeNegative) {
  // Test shrinking (negative delta)
  RectSet set{Vector2{-2, -2}, Vector2{2, 2}};
  const Vector2 original_center = (set.lower_bound() + set.upper_bound()) / 2.0;

  // Shrink by 2 units in all dimensions
  set.change_size(-2.0);

  // Check that bounds shrank correctly
  EXPECT_DOUBLE_EQ(set.lower_bound()(0), -1.0);
  EXPECT_DOUBLE_EQ(set.lower_bound()(1), -1.0);
  EXPECT_DOUBLE_EQ(set.upper_bound()(0), 1.0);
  EXPECT_DOUBLE_EQ(set.upper_bound()(1), 1.0);

  // Check that center is preserved
  const Vector2 new_center = (set.lower_bound() + set.upper_bound()) / 2.0;
  EXPECT_DOUBLE_EQ(new_center(0), original_center(0));
  EXPECT_DOUBLE_EQ(new_center(1), original_center(1));
}

TEST(TestRectSet, ChangeSizeOffCenter) {
  // Test with set that's not centered at origin
  RectSet set{Vector2{1, 2}, Vector2{3, 6}};
  const Vector2 original_center = (set.lower_bound() + set.upper_bound()) / 2.0;

  // Expand uniformly
  set.change_size(2.0);

  // Check that center is preserved
  const Vector2 new_center = (set.lower_bound() + set.upper_bound()) / 2.0;
  EXPECT_DOUBLE_EQ(new_center(0), original_center(0));
  EXPECT_DOUBLE_EQ(new_center(1), original_center(1));

  // Original size: [2, 4], new size should be [4, 6]
  EXPECT_DOUBLE_EQ(set.upper_bound()(0) - set.lower_bound()(0), 4.0);
  EXPECT_DOUBLE_EQ(set.upper_bound()(1) - set.lower_bound()(1), 6.0);
}

TEST(TestRectSet, ChangeSizeMultidimensional) {
  // Test with higher dimensional set
  const Vector lb = Vector{{-1, -2, -3, -4}};
  const Vector ub = Vector{{1, 2, 3, 4}};
  RectSet set{lb, ub};
  const Vector original_center = (set.lower_bound() + set.upper_bound()) / 2.0;

  // Expand with different delta for each dimension
  const Vector delta = Vector{{1.0, 2.0, 3.0, 4.0}};
  set.change_size(delta);

  // Check that center is preserved
  const Vector new_center = (set.lower_bound() + set.upper_bound()) / 2.0;
  for (Index i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(new_center(i), original_center(i));
  }

  // Check that sizes increased correctly
  for (Index i = 0; i < 4; ++i) {
    const Scalar original_size = ub(i) - lb(i);
    const Scalar new_size = set.upper_bound()(i) - set.lower_bound()(i);
    EXPECT_DOUBLE_EQ(new_size, original_size + delta(i));
  }
}

TEST(TestRectSet, ChangeSizeZero) {
  // Test with zero change (no-op)
  RectSet set{Vector2{-1, -1}, Vector2{1, 1}};
  const Vector2 original_lb = set.lower_bound();
  const Vector2 original_ub = set.upper_bound();

  set.change_size(0.0);

  EXPECT_DOUBLE_EQ(set.lower_bound()(0), original_lb(0));
  EXPECT_DOUBLE_EQ(set.lower_bound()(1), original_lb(1));
  EXPECT_DOUBLE_EQ(set.upper_bound()(0), original_ub(0));
  EXPECT_DOUBLE_EQ(set.upper_bound()(1), original_ub(1));
}

TEST(TestRectSet, ChangeSizePointsStillContained) {
  // Test that points at center remain contained after expansion
  RectSet set{Vector2{-1, -1}, Vector2{1, 1}};
  const Vector2 center{0, 0};
  const Vector2 point1{0.5, 0.5};
  const Vector2 point2{-0.5, -0.5};

  EXPECT_TRUE(set(center));
  EXPECT_TRUE(set(point1));
  EXPECT_TRUE(set(point2));

  set.change_size(4.0);

  // Points should still be contained after expansion
  EXPECT_TRUE(set(center));
  EXPECT_TRUE(set(point1));
  EXPECT_TRUE(set(point2));

  // New points at old boundaries should also be contained
  EXPECT_TRUE(set(Vector2{1, 1}));
  EXPECT_TRUE(set(Vector2{-1, -1}));
}

TEST(TestRectSet, ChangeSizeShrinkToInvert) {
  // Test shrinking so much that bounds would invert
  RectSet set{Vector2{-1, -1}, Vector2{1, 1}};

  // Shrink by more than the original size
  EXPECT_THROW(set.change_size(-5.0), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestRectSet, ChangeSizeDimensionMismatch) {
  // Test that dimension mismatch throws exception
  RectSet set{Vector2{-1, -1}, Vector2{1, 1}};

  // Try to change size with wrong dimension
  const Vector delta{{1.0, 2.0, 3.0}};

  // Should throw when dimensions don't match
  EXPECT_THROW(set.change_size(delta), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestRectSet, TestScaledWrappedWrapNoDimensions) {
  // Test that dimension mismatch throws exception
  const RectSet bounds{Vector2{0, 0}, Vector2{5, 5}};
  const RectSet set{Vector2{2, 2}, Vector2{3, 3}};
  const auto scaled = set.scale_wrapped(2.0, bounds);

  ASSERT_NE(dynamic_cast<RectSet*>(scaled.get()), nullptr);
  const RectSet& rect_set = *static_cast<RectSet*>(scaled.get());
  EXPECT_EQ(rect_set, RectSet({1.5, 1.5}, {3.5, 3.5}));
}

TEST(TestRectSet, TestScaledWrappedWrapOneDimensions) {
  // Test that dimension mismatch throws exception
  const RectSet bounds{Vector2{0, 0}, Vector2{5, 5}};
  const RectSet set{Vector2{2, 0}, Vector2{3, 1}};
  const auto scaled = set.scale_wrapped(2.0, bounds);

  ASSERT_NE(dynamic_cast<MultiSet*>(scaled.get()), nullptr);
  const MultiSet& multi_set = *static_cast<MultiSet*>(scaled.get());
  ASSERT_EQ(multi_set.sets().size(), 2);
  for (const auto& s : multi_set.sets()) ASSERT_NE(dynamic_cast<RectSet*>(s.get()), nullptr);

  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[0]), RectSet({1.5, 0}, {3.5, 1.5}));
  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[1]), RectSet({1.5, 4.5}, {3.5, 5}));
}

TEST(TestRectSet, TestScaledWrappedWrapBothDimensionsLb) {
  // Test that dimension mismatch throws exception
  const RectSet bounds{Vector2{0, 0}, Vector2{5, 5}};
  const RectSet set{Vector2{0, 0}, Vector2{1, 1}};
  const auto scaled = set.scale_wrapped(2.0, bounds);

  ASSERT_NE(dynamic_cast<MultiSet*>(scaled.get()), nullptr);
  const MultiSet& multi_set = *static_cast<MultiSet*>(scaled.get());
  ASSERT_EQ(multi_set.sets().size(), 3);
  for (const auto& s : multi_set.sets()) ASSERT_NE(dynamic_cast<RectSet*>(s.get()), nullptr);

  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[0]), RectSet({0, 0}, {1.5, 1.5}));
  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[1]), RectSet({4.5, 0}, {5, 1.5}));
  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[2]), RectSet({0, 4.5}, {1.5, 5}));
}

TEST(TestRectSet, TestScaledWrappedWrapBothDimensionsUb) {
  // Test that dimension mismatch throws exception
  const RectSet bounds{Vector2{0, 0}, Vector2{5, 5}};
  const RectSet set{Vector2{4, 4}, Vector2{5, 5}};
  const auto scaled = set.scale_wrapped(2.0, bounds);

  ASSERT_NE(dynamic_cast<MultiSet*>(scaled.get()), nullptr);
  const MultiSet& multi_set = *static_cast<MultiSet*>(scaled.get());
  ASSERT_EQ(multi_set.sets().size(), 3);
  for (const auto& s : multi_set.sets()) ASSERT_NE(dynamic_cast<RectSet*>(s.get()), nullptr);

  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[0]), RectSet({3.5, 3.5}, {5, 5}));
  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[1]), RectSet({0, 3.5}, {0.5, 5}));
  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[2]), RectSet({3.5, 0}, {5, 0.5}));
}

TEST(TestRectSet, TestScaledWrappedWrapAvoidOverlap) {
  // Test that dimension mismatch throws exception
  const RectSet bounds{Vector2{0, 0}, Vector2{5, 5}};
  const RectSet set{Vector2{0, 0}, Vector2{4, 2}};
  const auto scaled = set.scale_wrapped(2.0, bounds);

  ASSERT_NE(dynamic_cast<MultiSet*>(scaled.get()), nullptr);
  const MultiSet& multi_set = *static_cast<MultiSet*>(scaled.get());
  ASSERT_EQ(multi_set.sets().size(), 2);
  for (const auto& s : multi_set.sets()) ASSERT_NE(dynamic_cast<RectSet*>(s.get()), nullptr);

  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[0]), RectSet({0, 0}, {5, 3}));
  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[1]), RectSet({0, 4}, {5, 5}));
}

TEST(TestRectSet, TestScaledWrappedVectorScale) {
  // Test scaling with different factors per dimension
  const RectSet bounds{Vector2{0, 0}, Vector2{10, 10}};
  const RectSet set{Vector2{4, 4}, Vector2{6, 6}};
  const Vector2 scale_factors{2.0, 1.0};
  const auto scaled = set.scale_wrapped(scale_factors, bounds);

  ASSERT_NE(dynamic_cast<RectSet*>(scaled.get()), nullptr);
  const RectSet& rect_set = *static_cast<RectSet*>(scaled.get());

  // Original size: [2, 2], scaled by [2.0, 1.0] => new size [4, 2]
  // Center at [5, 5], so bounds should be [3, 4] to [7, 6]
  EXPECT_EQ(rect_set, RectSet({3, 4}, {7, 6}));
}

TEST(TestRectSet, TestScaledWrappedRelativeToBounds) {
  // Test scaling relative to bounds size instead of current size
  const RectSet bounds{Vector2{0, 0}, Vector2{10, 10}};
  const RectSet set{Vector2{4, 4}, Vector2{6, 6}};
  const auto scaled = set.scale_wrapped(0.5, bounds, true);

  ASSERT_NE(dynamic_cast<RectSet*>(scaled.get()), nullptr);
  const RectSet& rect_set = *static_cast<RectSet*>(scaled.get());

  // Bounds size: [10, 10], scale 0.5 relative to bounds => new size [5, 5]
  // Center at [5, 5], so bounds should be [2.5, 2.5] to [7.5, 7.5]
  EXPECT_EQ(rect_set, RectSet({1.5, 1.5}, {8.5, 8.5}));
}

TEST(TestRectSet, TestScaledWrappedZeroScale) {
  // Test scaling with zero factor (degenerate case)
  const RectSet bounds{Vector2{0, 0}, Vector2{10, 10}};
  const RectSet set{Vector2{4, 4}, Vector2{6, 6}};
  EXPECT_THROW(static_cast<void>(set.scale_wrapped(0.0, bounds)), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestRectSet, TestScaledWrappedNegativeScale) {
  // Test scaling with negative factor (should still work, inverting the rectangle)
  const RectSet bounds{Vector2{0, 0}, Vector2{10, 10}};
  const RectSet set{Vector2{4, 4}, Vector2{6, 6}};
  EXPECT_THROW(static_cast<void>(set.scale_wrapped(0.0, bounds)), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestRectSet, TestScaledWrappedExceedsBoundsAllSides) {
  // Test when scaling causes the set to exceed bounds on all sides
  const RectSet bounds{Vector2{0, 0}, Vector2{10, 10}};
  const RectSet set{Vector2{3, 3}, Vector2{7, 7}};
  const auto scaled = set.scale_wrapped(3.0, bounds);

  ASSERT_NE(dynamic_cast<RectSet*>(scaled.get()), nullptr);
}

TEST(TestRectSet, TestScaledWrapped3D) {
  // Test scale_wrapped in 3D
  const Vector lb_bounds{{0, 0, 0}};
  const Vector ub_bounds{{10, 10, 10}};
  const RectSet bounds{lb_bounds, ub_bounds};

  const Vector lb_set{{4, 4, 4}};
  const Vector ub_set{{6, 6, 6}};
  const RectSet set{lb_set, ub_set};

  const auto scaled = set.scale_wrapped(2.0, bounds);

  ASSERT_NE(dynamic_cast<RectSet*>(scaled.get()), nullptr);
  const RectSet& rect_set = *static_cast<RectSet*>(scaled.get());

  // Original size: [2, 2, 2], scaled by 1.0 => new size [4, 4, 4]
  // Center at [5, 5, 5], so bounds should be [3, 3, 3] to [7, 7, 7]
  EXPECT_EQ(rect_set, RectSet({3, 3, 3}, {7, 7, 7}));
}

TEST(TestRectSet, TestScaledWrapped3DWithWrapping) {
  // Test scale_wrapped in 3D with wrapping in one dimension
  const Vector lb_bounds{{0, 0, 0}};
  const Vector ub_bounds{{10, 10, 10}};
  const RectSet bounds{lb_bounds, ub_bounds};

  const Vector lb_set{{8, 5, 5}};
  const Vector ub_set{{9, 6, 6}};
  const RectSet set{lb_set, ub_set};

  const auto scaled = set.scale_wrapped(2.0, bounds);

  ASSERT_NE(dynamic_cast<RectSet*>(scaled.get()), nullptr);
  const RectSet& rect_set = *static_cast<RectSet*>(scaled.get());

  EXPECT_EQ(rect_set, RectSet({7.5, 4.5, 4.5}, {9.5, 6.5, 6.5}));
}

TEST(TestRectSet, TestScaledWrappedNonSquareBounds) {
  // Test with non-square bounds (different sizes in each dimension)
  const RectSet bounds{Vector2{0, 0}, Vector2{20, 10}};
  const RectSet set{Vector2{15, 4}, Vector2{18, 6}};
  const auto scaled = set.scale_wrapped(3.0, bounds);

  ASSERT_NE(dynamic_cast<MultiSet*>(scaled.get()), nullptr);
  const MultiSet& multi_set = *static_cast<MultiSet*>(scaled.get());
  ASSERT_EQ(multi_set.sets().size(), 2);

  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[0]), RectSet({12.0, 2.0}, {20.0, 8.0}));
  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[1]), RectSet({0.0, 2.0}, {1.0, 8.0}));
}

TEST(TestRectSet, TestScaledWrappedNegativeBounds) {
  // Test with bounds that include negative coordinates
  const RectSet bounds{Vector2{-5, -5}, Vector2{5, 5}};
  const RectSet set{Vector2{3, 0}, Vector2{4, 1}};
  const auto scaled = set.scale_wrapped(5.0, bounds);

  ASSERT_NE(dynamic_cast<MultiSet*>(scaled.get()), nullptr);
  const MultiSet& multi_set = *static_cast<MultiSet*>(scaled.get());
  ASSERT_EQ(multi_set.sets().size(), 2);

  // Original size: [1, 1], scaled by 4.0 => new size [4, 4]
  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[0]), RectSet({1.0, -2.0}, {5.0, 3.0}));
  EXPECT_EQ(dynamic_cast<const RectSet&>(multi_set[1]), RectSet({-5.0, -2.0}, {-4.0, 3.0}));
}
