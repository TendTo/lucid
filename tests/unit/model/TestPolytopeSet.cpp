/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/PolytopeSet.h"
#include "lucid/util/exception.h"

using lucid::Index;
using lucid::Matrix;
using lucid::PolytopeSet;
using lucid::Scalar;
using lucid::Set;
using lucid::Vector;
using lucid::Vector2;
using lucid::Vector3;
using lucid::VectorI;

// Test basic construction and properties
TEST(TestPolytopeSet, Construction) {
  // Create a simple 2D unit square: -1 <= x <= 1, -1 <= y <= 1
  Matrix A(4, 2);
  A << 1, 0,  // x <= 1
      -1, 0,  // x >= -1 (i.e., -x <= 1)
      0, 1,   // y <= 1
      0, -1;  // y >= -1 (i.e., -y <= 1)
  Vector b(4);
  b << 1, 1, 1, 1;

  const PolytopeSet polytope{A, b};

  EXPECT_EQ(polytope.dimension(), 2);
  EXPECT_EQ(polytope.A().rows(), 4);
  EXPECT_EQ(polytope.A().cols(), 2);
  EXPECT_EQ(polytope.b().size(), 4);
  EXPECT_TRUE(polytope.A().isApprox(A));
  EXPECT_TRUE(polytope.b().isApprox(b));
}

TEST(TestPolytopeSet, ConstructionInitializerList) {
  // Create a simple triangle: x >= 0, y >= 0, x + y <= 1
  const PolytopeSet polytope{
      {{-1, 0}, {0, -1}, {1, 1}},  // A matrix: [-1,0], [0,-1], [1,1]
      {0, 0, 1}                    // b vector: [0, 0, 1]
  };

  EXPECT_EQ(polytope.dimension(), 2);
  EXPECT_EQ(polytope.A().rows(), 3);
  EXPECT_EQ(polytope.A().cols(), 2);
  EXPECT_EQ(polytope.b().size(), 3);
}

TEST(TestPolytopeSet, FromBox2D) {
  const std::vector<std::pair<Scalar, Scalar>> bounds = {{-2.0, 3.0}, {-1.0, 4.0}};
  const PolytopeSet polytope = PolytopeSet::from_box(bounds);

  EXPECT_EQ(polytope.dimension(), 2);
  EXPECT_EQ(polytope.A().rows(), 4);  // 2 * dimension
  EXPECT_EQ(polytope.A().cols(), 2);
  EXPECT_EQ(polytope.b().size(), 4);

  // Test that the box constraints are correctly represented
  // Expected A matrix:
  // [ 1,  0]  x <= 3
  // [ 0,  1]  y <= 4
  // [-1,  0]  x >= -2 (i.e., -x <= 2)
  // [ 0, -1]  y >= -1 (i.e., -y <= 1)
  Matrix expected_A(4, 2);
  expected_A << 1, 0, 0, 1, -1, 0, 0, -1;
  Vector expected_b(4);
  expected_b << 3, 4, 2, 1;

  EXPECT_TRUE(polytope.A().isApprox(expected_A));
  EXPECT_TRUE(polytope.b().isApprox(expected_b));
}

TEST(TestPolytopeSet, FromBox3D) {
  const std::vector<std::pair<Scalar, Scalar>> bounds = {{0.0, 1.0}, {-1.0, 2.0}, {-0.5, 0.5}};
  const PolytopeSet polytope = PolytopeSet::from_box(bounds);

  EXPECT_EQ(polytope.dimension(), 3);
  EXPECT_EQ(polytope.A().rows(), 6);  // 2 * dimension
  EXPECT_EQ(polytope.A().cols(), 3);
  EXPECT_EQ(polytope.b().size(), 6);
}

// Test containment functionality
TEST(TestPolytopeSet, ContainsUnitSquare) {
  // Unit square: -1 <= x <= 1, -1 <= y <= 1
  const PolytopeSet polytope = PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}});

  // Points inside the polytope
  EXPECT_TRUE(polytope(Vector2{0, 0}));       // Center
  EXPECT_TRUE(polytope(Vector2{0.5, 0.5}));   // Inside
  EXPECT_TRUE(polytope(Vector2{-0.5, 0.5}));  // Inside

  // Points on the boundary
  EXPECT_TRUE(polytope(Vector2{1, 0}));    // Right edge
  EXPECT_TRUE(polytope(Vector2{-1, 0}));   // Left edge
  EXPECT_TRUE(polytope(Vector2{0, 1}));    // Top edge
  EXPECT_TRUE(polytope(Vector2{0, -1}));   // Bottom edge
  EXPECT_TRUE(polytope(Vector2{1, 1}));    // Corner
  EXPECT_TRUE(polytope(Vector2{-1, -1}));  // Corner

  // Points outside the polytope
  EXPECT_FALSE(polytope(Vector2{1.1, 0}));   // Outside right
  EXPECT_FALSE(polytope(Vector2{-1.1, 0}));  // Outside left
  EXPECT_FALSE(polytope(Vector2{0, 1.1}));   // Outside top
  EXPECT_FALSE(polytope(Vector2{0, -1.1}));  // Outside bottom
  EXPECT_FALSE(polytope(Vector2{2, 2}));     // Far outside
}

TEST(TestPolytopeSet, ContainsTriangle) {
  // Triangle: x >= 0, y >= 0, x + y <= 1
  const PolytopeSet polytope{
      {{-1, 0}, {0, -1}, {1, 1}},  // A matrix
      {0, 0, 1}                    // b vector
  };

  // Points inside the triangle
  EXPECT_TRUE(polytope(Vector2{0, 0}));        // Corner (origin)
  EXPECT_TRUE(polytope(Vector2{0.25, 0.25}));  // Inside
  EXPECT_TRUE(polytope(Vector2{0.5, 0.25}));   // Inside

  // Points on the boundary
  EXPECT_TRUE(polytope(Vector2{1, 0}));      // Corner
  EXPECT_TRUE(polytope(Vector2{0, 1}));      // Corner
  EXPECT_TRUE(polytope(Vector2{0.5, 0.5}));  // On hypotenuse

  // Points outside the triangle
  EXPECT_FALSE(polytope(Vector2{-0.1, 0}));   // Outside x >= 0
  EXPECT_FALSE(polytope(Vector2{0, -0.1}));   // Outside y >= 0
  EXPECT_FALSE(polytope(Vector2{0.6, 0.6}));  // Outside x + y <= 1
  EXPECT_FALSE(polytope(Vector2{2, 2}));      // Far outside
}

TEST(TestPolytopeSet, Contains3D) {
  // 3D unit cube: 0 <= x,y,z <= 1
  const PolytopeSet polytope = PolytopeSet::from_box({{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}});

  // Points inside the cube
  EXPECT_TRUE(polytope(Vector3{0.5, 0.5, 0.5}));    // Center
  EXPECT_TRUE(polytope(Vector3{0.25, 0.75, 0.1}));  // Inside

  // Points on the boundary
  EXPECT_TRUE(polytope(Vector3{0, 0, 0}));      // Corner
  EXPECT_TRUE(polytope(Vector3{1, 1, 1}));      // Corner
  EXPECT_TRUE(polytope(Vector3{0.5, 0.5, 1}));  // Face

  // Points outside the cube
  EXPECT_FALSE(polytope(Vector3{-0.1, 0.5, 0.5}));  // Outside
  EXPECT_FALSE(polytope(Vector3{1.1, 0.5, 0.5}));   // Outside
  EXPECT_FALSE(polytope(Vector3{2, 2, 2}));         // Far outside
}

// Test scaling functionality
TEST(TestPolytopeSet, Scale) {
  // Unit square: -1 <= x <= 1, -1 <= y <= 1
  PolytopeSet polytope = PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}});

  // Scale by factor of 2
  polytope.scale(2.0);

  // Original b vector was [1, 1, 1, 1], after scaling should be [2, 2, 2, 2]
  Vector expected_b(4);
  expected_b << 2, 2, 2, 2;
  EXPECT_TRUE(polytope.b().isApprox(expected_b));

  // Now the polytope should represent -2 <= x <= 2, -2 <= y <= 2
  EXPECT_TRUE(polytope(Vector2{0, 0}));      // Center
  EXPECT_TRUE(polytope(Vector2{2, 0}));      // New boundary
  EXPECT_TRUE(polytope(Vector2{-2, 0}));     // New boundary
  EXPECT_FALSE(polytope(Vector2{2.1, 0}));   // Outside new boundary
  EXPECT_FALSE(polytope(Vector2{-2.1, 0}));  // Outside new boundary
}

TEST(TestPolytopeSet, ScaleNegative) {
  // Unit square
  PolytopeSet polytope = PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}});

  // Scale by negative factor
  polytope.scale(-0.5);

  Vector expected_b(4);
  expected_b << -0.5, -0.5, -0.5, -0.5;
  EXPECT_TRUE(polytope.b().isApprox(expected_b));
}

TEST(TestPolytopeSet, ScaleZero) {
  // Unit square
  PolytopeSet polytope = PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}});

  polytope.scale(0.0);

  Vector expected_b = Vector::Zero(4);
  EXPECT_TRUE(polytope.b().isApprox(expected_b));
}

// Test dimension consistency
TEST(TestPolytopeSet, Dimension) {
  // 1D polytope
  const PolytopeSet polytope1d = PolytopeSet::from_box({{-1.0, 1.0}});
  EXPECT_EQ(polytope1d.dimension(), 1);

  // 2D polytope
  const PolytopeSet polytope2d = PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}});
  EXPECT_EQ(polytope2d.dimension(), 2);

  // 3D polytope
  const PolytopeSet polytope3d = PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}});
  EXPECT_EQ(polytope3d.dimension(), 3);

  // High-dimensional polytope
  std::vector<std::pair<Scalar, Scalar>> bounds_10d(10, {-1.0, 1.0});
  const PolytopeSet polytope10d = PolytopeSet::from_box(bounds_10d);
  EXPECT_EQ(polytope10d.dimension(), 10);
}

// Test lattice generation
TEST(TestPolytopeSet, Lattice2DInclude) {
  // Unit square: -1 <= x <= 1, -1 <= y <= 1
  const PolytopeSet polytope = PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}});
  const VectorI points_per_dim = VectorI::Constant(2, 3);
  const Matrix lattice = polytope.lattice(points_per_dim, true);

  EXPECT_EQ(lattice.cols(), 2);
  // All lattice points should be contained in the polytope
  for (Index i = 0; i < lattice.rows(); ++i) {
    EXPECT_TRUE(polytope(lattice.row(i)));
  }
}

TEST(TestPolytopeSet, Lattice2D) {
  // Unit square: -1 <= x <= 1, -1 <= y <= 1
  const PolytopeSet polytope = PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}});
  const VectorI points_per_dim = VectorI::Constant(2, 3);
  const Matrix lattice = polytope.lattice(points_per_dim, false);

  EXPECT_EQ(lattice.cols(), 2);
  // All lattice points should be contained in the polytope
  for (Index i = 0; i < lattice.rows(); ++i) {
    EXPECT_TRUE(polytope(lattice.row(i)));
  }
}

TEST(TestPolytopeSet, LatticeTriangle) {
  // Triangle: x >= 0, y >= 0, x + y <= 1
  const PolytopeSet polytope{
      {{-1, 0}, {0, -1}, {1, 1}},  // A matrix
      {0, 0, 1}                    // b vector
  };

  const VectorI points_per_dim = VectorI::Constant(2, 5);
  const Matrix lattice = polytope.lattice(points_per_dim, true);

  EXPECT_EQ(lattice.cols(), 2);
  // All lattice points should be contained in the polytope
  for (Index i = 0; i < lattice.rows(); ++i) {
    EXPECT_TRUE(polytope(lattice.row(i))) << "Point (" << lattice(i, 0) << ", " << lattice(i, 1) << ") not in polytope";
  }

  // Should have fewer points than a full grid since we're filtering out points outside the triangle
  EXPECT_LT(lattice.rows(), 25);  // 5*5 = 25 would be the full grid
}

TEST(TestPolytopeSet, Lattice3D) {
  // 3D unit cube: 0 <= x,y,z <= 1
  const PolytopeSet polytope = PolytopeSet::from_box({{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}});
  const VectorI points_per_dim = VectorI::Constant(3, 3);
  const Matrix lattice = polytope.lattice(points_per_dim, true);

  EXPECT_EQ(lattice.cols(), 3);
  EXPECT_EQ(lattice.rows(), 27);  // 3^3 = 27 points for a cube

  // All lattice points should be contained in the polytope
  for (Index i = 0; i < lattice.rows(); ++i) {
    EXPECT_TRUE(polytope(lattice.row(i)));
  }
}

// Test exception handling
TEST(TestPolytopeSet, InvalidArguments) {
  // Test mismatched A and b dimensions
  Matrix A(3, 2);
  A << 1, 0, 0, 1, 1, 1;
  Vector b(2);  // Wrong size - should be 3
  b << 1, 1;

  EXPECT_THROW(PolytopeSet(A, b), lucid::exception::LucidInvalidArgumentException);

  // Test empty matrix
  Matrix empty_A(0, 2);
  Vector empty_b(0);
  EXPECT_THROW(PolytopeSet(empty_A, empty_b), lucid::exception::LucidInvalidArgumentException);

  // Test zero columns
  Matrix zero_cols_A(2, 0);
  Vector zero_cols_b(2);
  zero_cols_b << 1, 1;
  EXPECT_THROW(PolytopeSet(zero_cols_A, zero_cols_b), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestPolytopeSet, InvalidFromBox) {
  // Test empty bounds
  std::vector<std::pair<Scalar, Scalar>> empty_bounds;
  EXPECT_THROW(PolytopeSet::from_box(empty_bounds), lucid::exception::LucidInvalidArgumentException);

  // Test invalid bounds (lower > upper)
  std::vector<std::pair<Scalar, Scalar>> invalid_bounds = {{2.0, 1.0}};
  EXPECT_THROW(PolytopeSet::from_box(invalid_bounds), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestPolytopeSet, InvalidInitializerList) {
  // Test inconsistent row sizes in initializer list
  EXPECT_THROW(PolytopeSet({{1, 0}, {0, 1, 2}}, {1, 1}), lucid::exception::LucidInvalidArgumentException);

  // Test mismatched dimensions between A and b in initializer list
  EXPECT_THROW(PolytopeSet({{1, 0}, {0, 1}}, {1}), lucid::exception::LucidInvalidArgumentException);
}

// Test contains with wrong dimensions
TEST(TestPolytopeSet, WrongDimensionContains) {
  const PolytopeSet polytope2d = PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}});
  const PolytopeSet polytope3d = PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}});

  // Test checking points with wrong dimensions
  EXPECT_THROW(static_cast<void>(polytope2d(Vector3{0, 0, 0})), lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(static_cast<void>(polytope3d(Vector2{0, 0})), lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(static_cast<void>(polytope2d(Vector::Constant(1, 0.0))),
               lucid::exception::LucidInvalidArgumentException);
}

// Test mathematical properties
TEST(TestPolytopeSet, MathematicalProperties) {
  // Create a simple 2D polytope: x >= 0, y >= 0, x + y <= 2
  Matrix A(3, 2);
  A << -1, 0,  // x >= 0 (i.e., -x <= 0)
      0, -1,   // y >= 0 (i.e., -y <= 0)
      1, 1;    // x + y <= 2
  Vector b(3);
  b << 0, 0, 2;

  const PolytopeSet polytope{A, b};

  // Test corner points
  EXPECT_TRUE(polytope(Vector2{0, 0}));  // Origin
  EXPECT_TRUE(polytope(Vector2{2, 0}));  // (2,0)
  EXPECT_TRUE(polytope(Vector2{0, 2}));  // (0,2)
  EXPECT_TRUE(polytope(Vector2{1, 1}));  // (1,1) on the line

  // Test points just outside
  EXPECT_FALSE(polytope(Vector2{-0.001, 0}));     // Just outside x >= 0
  EXPECT_FALSE(polytope(Vector2{0, -0.001}));     // Just outside y >= 0
  EXPECT_FALSE(polytope(Vector2{1.001, 1.001}));  // Just outside x + y <= 2

  // Test interior points
  EXPECT_TRUE(polytope(Vector2{0.5, 0.5}));
  EXPECT_TRUE(polytope(Vector2{1, 0.5}));
  EXPECT_TRUE(polytope(Vector2{0.5, 1}));
}

// Test edge cases with very small and very large values
TEST(TestPolytopeSet, EdgeCases) {
  // Very small polytope
  const PolytopeSet tiny = PolytopeSet::from_box({{-1e-8, 1e-8}, {-1e-8, 1e-8}});
  EXPECT_TRUE(tiny(Vector2{0, 0}));
  EXPECT_FALSE(tiny(Vector2{1e-7, 0}));

  // Very large polytope
  const PolytopeSet large = PolytopeSet::from_box({{-1e6, 1e6}, {-1e6, 1e6}});
  EXPECT_TRUE(large(Vector2{0, 0}));
  EXPECT_TRUE(large(Vector2{1e5, 1e5}));
  EXPECT_FALSE(large(Vector2{2e6, 0}));
}

// Test precision around boundary
TEST(TestPolytopeSet, BoundaryPrecision) {
  const PolytopeSet polytope = PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}});

  // Test points very close to the boundary
  const double eps = std::numeric_limits<double>::epsilon();

  // Points just inside (should be contained due to tolerance)
  EXPECT_TRUE(polytope(Vector2{1.0 - eps, 0}));
  EXPECT_TRUE(polytope(Vector2{-1.0 + eps, 0}));

  // Points exactly on boundary
  EXPECT_TRUE(polytope(Vector2{1.0, 0}));
  EXPECT_TRUE(polytope(Vector2{-1.0, 0}));
}

// Test with 1D polytope (interval)
TEST(TestPolytopeSet, OneDimensional) {
  // Interval [0, 2]
  const PolytopeSet interval = PolytopeSet::from_box({{0.0, 2.0}});

  EXPECT_EQ(interval.dimension(), 1);

  // Test containment
  EXPECT_TRUE(interval(Vector::Constant(1, 0.0)));    // 0
  EXPECT_TRUE(interval(Vector::Constant(1, 1.0)));    // 1
  EXPECT_TRUE(interval(Vector::Constant(1, 2.0)));    // 2
  EXPECT_FALSE(interval(Vector::Constant(1, -0.1)));  // -0.1
  EXPECT_FALSE(interval(Vector::Constant(1, 2.1)));   // 2.1
}

// Test polymorphic behavior (inheritance from Set)
TEST(TestPolytopeSet, PolymorphicBehavior) {
  std::unique_ptr<Set> set = std::make_unique<PolytopeSet>(PolytopeSet::from_box({{-1.0, 1.0}, {-1.0, 1.0}}));

  EXPECT_EQ(set->dimension(), 2);
  EXPECT_TRUE(set->contains(Vector2{0, 0}));
  EXPECT_FALSE(set->contains(Vector2{2, 2}));

  // Test single sample
  EXPECT_THROW(const Vector sample = set->sample(), lucid::exception::LucidNotImplementedException);
  // EXPECT_EQ(sample.size(), 2);
  // EXPECT_TRUE(set->contains(sample));
}

// Test high-dimensional polytope
TEST(TestPolytopeSet, HighDimensional) {
  // 5D hypercube: -1 <= x_i <= 1 for i = 1, ..., 5
  std::vector<std::pair<Scalar, Scalar>> bounds_5d(5, {-1.0, 1.0});
  const PolytopeSet polytope = PolytopeSet::from_box(bounds_5d);

  EXPECT_EQ(polytope.dimension(), 5);
  EXPECT_EQ(polytope.A().rows(), 10);  // 2 * 5
  EXPECT_EQ(polytope.A().cols(), 5);

  // Test center point
  const Vector center = Vector::Zero(5);
  EXPECT_TRUE(polytope(center));

  // Test corner points
  const Vector corner_pos = Vector::Ones(5);
  const Vector corner_neg = -Vector::Ones(5);
  EXPECT_TRUE(polytope(corner_pos));
  EXPECT_TRUE(polytope(corner_neg));

  // Test outside point
  Vector outside = Vector::Ones(5) * 1.1;
  EXPECT_FALSE(polytope(outside));
}

// Test degenerate cases
TEST(TestPolytopeSet, DegenerateCases) {
  // Line segment in 2D: x = 0, 0 <= y <= 1
  Matrix A(4, 2);
  A << 1, 0,  // x <= 0
      -1, 0,  // x >= 0 (i.e., -x <= 0), combined with above gives x = 0
      0, 1,   // y <= 1
      0, -1;  // y >= 0 (i.e., -y <= 0)
  Vector b(4);
  b << 0, 0, 1, 0;

  const PolytopeSet line{A, b};

  EXPECT_EQ(line.dimension(), 2);

  // Points on the line segment
  EXPECT_TRUE(line(Vector2{0, 0}));
  EXPECT_TRUE(line(Vector2{0, 0.5}));
  EXPECT_TRUE(line(Vector2{0, 1}));

  // Points off the line
  EXPECT_FALSE(line(Vector2{0.001, 0.5}));
  EXPECT_FALSE(line(Vector2{-0.001, 0.5}));
  EXPECT_FALSE(line(Vector2{0, 1.001}));
  EXPECT_FALSE(line(Vector2{0, -0.001}));
}

// Test empty polytope (infeasible constraints)
TEST(TestPolytopeSet, EmptyPolytope) {
  // Contradictory constraints: x <= -1 and x >= 1
  Matrix A(2, 1);
  A << 1,  // x <= -1
      -1;  // x >= 1 (i.e., -x <= -1)
  Vector b(2);
  b << -1, -1;

  const PolytopeSet empty{A, b};

  EXPECT_EQ(empty.dimension(), 1);

  // No point should be in this polytope
  EXPECT_FALSE(empty(Vector::Constant(1, 0.0)));
  EXPECT_FALSE(empty(Vector::Constant(1, 1.0)));
  EXPECT_FALSE(empty(Vector::Constant(1, -1.0)));
}

TEST(TestPolytopeSet, EmptyPolytopeSampling) {
  // Contradictory constraints: x <= -1 and x >= 1
  Matrix A(2, 1);
  A << 1,  // x <= -1
      -1;  // x >= 1 (i.e., -x <= -1)
  Vector b(2);
  b << -1, -1;

  const PolytopeSet empty{A, b};
  EXPECT_ANY_THROW(static_cast<void>(empty.lattice(1, true)));
}

TEST(TestPolytopeSet, UnboundedPolytopeSampling) {
  // Contradictory constraints: x <= -1 and x >= 1
  Matrix A(2, 2);
  A << 2, 3,  // 2x + 2y <= 2
      1, -4;  // x - 4y <= 3
  Vector b(2);
  b << 2, 3;

  const PolytopeSet unbounded{A, b};
  EXPECT_ANY_THROW(static_cast<void>(unbounded.lattice(1, true)));
}

// Test single point polytope
TEST(TestPolytopeSet, SinglePoint) {
  // Point (1, 2): x = 1, y = 2
  Matrix A(4, 2);
  A << 1, 0,  // x <= 1
      -1, 0,  // x >= 1 (i.e., -x <= -1)
      0, 1,   // y <= 2
      0, -1;  // y >= 2 (i.e., -y <= -2)
  Vector b(4);
  b << 1, -1, 2, -2;

  const PolytopeSet point{A, b};

  EXPECT_EQ(point.dimension(), 2);

  // Only the point (1, 2) should be contained
  EXPECT_TRUE(point(Vector2{1, 2}));

  // Any other point should not be contained
  EXPECT_FALSE(point(Vector2{1.001, 2}));
  EXPECT_FALSE(point(Vector2{1, 2.001}));
  EXPECT_FALSE(point(Vector2{0.999, 2}));
  EXPECT_FALSE(point(Vector2{1, 1.999}));
  EXPECT_FALSE(point(Vector2{0, 0}));
}

// Test matrix/vector accessors
TEST(TestPolytopeSet, Accessors) {
  Matrix A(2, 2);
  A << 1, 0, 0, 1;
  Vector b(2);
  b << 5, 3;

  const PolytopeSet polytope{A, b};

  // Test const accessors
  EXPECT_TRUE(polytope.A().isApprox(A));
  EXPECT_TRUE(polytope.b().isApprox(b));

  // Ensure returned references are const-correct
  const Matrix& A_ref = polytope.A();
  const Vector& b_ref = polytope.b();
  EXPECT_TRUE(A_ref.isApprox(A));
  EXPECT_TRUE(b_ref.isApprox(b));
}
