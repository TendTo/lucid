/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/RectSet.h"

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
    for (Index row = 0; row < x.rows(); row++) EXPECT_TRUE(set(x.row(row)));
  }
}

TEST(TestRectSet, LatticeNoEndpointsSamePointsPerDimension) {
  constexpr int points_per_dim = 3;
  const RectSet set{Vector2{-1, -1}, Vector2{1, 1}, 0};
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
  const RectSet set{Vector2{-1, -1}, Vector2{1, 1}, 0};
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
