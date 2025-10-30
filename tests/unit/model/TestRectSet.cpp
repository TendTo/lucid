/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/RectSet.h"
#include "lucid/util/exception.h"

using lucid::Index;
using lucid::Matrix;
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
