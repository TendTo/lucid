/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/EllipseSet.h"
#include "lucid/util/exception.h"
#include "lucid/util/random.h"

using lucid::EllipseSet;
using lucid::Index;
using lucid::Matrix;
using lucid::Scalar;
using lucid::Set;
using lucid::Vector;
using lucid::Vector2;
using lucid::Vector3;
using lucid::VectorI;

// Test basic construction and contains functionality
TEST(TestEllipseSet, ConstructionWithRadiiVector) {
  const Vector2 center{1.5, -2.5};
  const Vector2 radii{3.0, 2.0};
  const EllipseSet ellipse{center, radii};

  EXPECT_EQ(ellipse.dimension(), 2);
  EXPECT_EQ(ellipse.center(), center);
  EXPECT_EQ(ellipse.radii(), radii);
}

TEST(TestEllipseSet, ConstructionWithUniformRadius) {
  const Vector2 center{1.5, -2.5};
  const Scalar radius = 3.0;
  const EllipseSet ellipse{center, radius};

  EXPECT_EQ(ellipse.dimension(), 2);
  EXPECT_EQ(ellipse.center(), center);
  EXPECT_EQ(ellipse.radii(), Vector2::Constant(radius));
}

TEST(TestEllipseSet, Contains2DCircle) {
  // Test with equal radii (circle)
  const Vector2 center{0, 0};
  const Vector2 radii{1.0, 1.0};
  const EllipseSet ellipse{center, radii};

  // Points inside the ellipse
  EXPECT_TRUE(ellipse(Vector2{0, 0}));      // Center
  EXPECT_TRUE(ellipse(Vector2{0.5, 0.5}));  // Inside
  EXPECT_TRUE(ellipse(Vector2{0.7, 0.7}));  // Close to boundary but inside

  // Points on the boundary
  EXPECT_TRUE(ellipse(Vector2{1, 0}));   // Boundary
  EXPECT_TRUE(ellipse(Vector2{0, 1}));   // Boundary
  EXPECT_TRUE(ellipse(Vector2{-1, 0}));  // Boundary
  EXPECT_TRUE(ellipse(Vector2{0, -1}));  // Boundary

  // Points outside the ellipse
  EXPECT_FALSE(ellipse(Vector2{1.1, 0}));    // Outside
  EXPECT_FALSE(ellipse(Vector2{0, 1.1}));    // Outside
  EXPECT_FALSE(ellipse(Vector2{1.5, 1.5}));  // Far outside
  EXPECT_FALSE(ellipse(Vector2{-2, -2}));    // Far outside
}

TEST(TestEllipseSet, Contains2DEllipse) {
  // Test with different radii (actual ellipse)
  const Vector2 center{0, 0};
  const Vector2 radii{2.0, 1.0};  // Wider in x, narrower in y
  const EllipseSet ellipse{center, radii};

  // Points inside the ellipse
  EXPECT_TRUE(ellipse(Vector2{0, 0}));      // Center
  EXPECT_TRUE(ellipse(Vector2{1.0, 0}));    // Inside
  EXPECT_TRUE(ellipse(Vector2{0, 0.5}));    // Inside
  EXPECT_TRUE(ellipse(Vector2{1.5, 0.5}));  // Inside

  // Points on the boundary
  EXPECT_TRUE(ellipse(Vector2{2, 0}));   // Boundary along major axis
  EXPECT_TRUE(ellipse(Vector2{-2, 0}));  // Boundary along major axis
  EXPECT_TRUE(ellipse(Vector2{0, 1}));   // Boundary along minor axis
  EXPECT_TRUE(ellipse(Vector2{0, -1}));  // Boundary along minor axis

  // Points outside the ellipse
  EXPECT_FALSE(ellipse(Vector2{2.1, 0}));  // Outside major axis
  EXPECT_FALSE(ellipse(Vector2{0, 1.1}));  // Outside minor axis
  EXPECT_FALSE(ellipse(Vector2{2, 1}));    // Outside (both coordinates at max)
  EXPECT_FALSE(ellipse(Vector2{1.5, 1}));  // Outside
}

TEST(TestEllipseSet, Contains3DEllipsoid) {
  const Vector3 center{1, 2, 3};
  const Vector3 radii{3.0, 2.0, 1.0};
  const EllipseSet ellipse{center, radii};

  // Points inside the ellipsoid
  EXPECT_TRUE(ellipse(Vector3{1, 2, 3}));    // Center
  EXPECT_TRUE(ellipse(Vector3{2, 2, 3}));    // Inside (1 unit in x direction)
  EXPECT_TRUE(ellipse(Vector3{1, 3, 3}));    // Inside (1 unit in y direction)
  EXPECT_TRUE(ellipse(Vector3{1, 2, 3.5}));  // Inside (0.5 units in z direction)

  // Points on the boundary
  EXPECT_TRUE(ellipse(Vector3{4, 2, 3}));   // Boundary (3 units in x)
  EXPECT_TRUE(ellipse(Vector3{-2, 2, 3}));  // Boundary (-3 units in x)
  EXPECT_TRUE(ellipse(Vector3{1, 4, 3}));   // Boundary (2 units in y)
  EXPECT_TRUE(ellipse(Vector3{1, 0, 3}));   // Boundary (-2 units in y)
  EXPECT_TRUE(ellipse(Vector3{1, 2, 4}));   // Boundary (1 unit in z)
  EXPECT_TRUE(ellipse(Vector3{1, 2, 2}));   // Boundary (-1 unit in z)

  // Points outside the ellipsoid
  EXPECT_FALSE(ellipse(Vector3{5, 2, 3}));  // Outside (4 units in x)
  EXPECT_FALSE(ellipse(Vector3{1, 5, 3}));  // Outside (3 units in y)
  EXPECT_FALSE(ellipse(Vector3{1, 2, 5}));  // Outside (2 units in z)
  EXPECT_FALSE(ellipse(Vector3{4, 4, 4}));  // Far outside
}

TEST(TestEllipseSet, ContainsOffCenter) {
  const Vector2 center{-2, 3};
  const Vector2 radii{2.0, 1.5};
  const EllipseSet ellipse{center, radii};

  // Points inside the ellipse
  EXPECT_TRUE(ellipse(Vector2{-2, 3}));    // Center
  EXPECT_TRUE(ellipse(Vector2{-1, 3}));    // Inside
  EXPECT_TRUE(ellipse(Vector2{-2, 3.5}));  // Inside

  // Points on the boundary
  EXPECT_TRUE(ellipse(Vector2{0, 3}));     // Boundary (2 units in x)
  EXPECT_TRUE(ellipse(Vector2{-4, 3}));    // Boundary (-2 units in x)
  EXPECT_TRUE(ellipse(Vector2{-2, 4.5}));  // Boundary (1.5 units in y)
  EXPECT_TRUE(ellipse(Vector2{-2, 1.5}));  // Boundary (-1.5 units in y)

  // Points outside the ellipse
  EXPECT_FALSE(ellipse(Vector2{1, 3}));   // Outside (3 units in x)
  EXPECT_FALSE(ellipse(Vector2{-2, 5}));  // Outside (2 units in y)
}

TEST(TestEllipseSet, ZeroRadii) {
  // Test with zero radius in all dimensions (point set)
  const Vector2 center{1, 1};
  const Vector2 radii{0.0, 0.0};
  const EllipseSet ellipse{center, radii};

  // Only the center point should be contained
  EXPECT_TRUE(ellipse(Vector2{1, 1}));

  // Any other point should not be contained
  EXPECT_FALSE(ellipse(Vector2{1.001, 1}));
  EXPECT_FALSE(ellipse(Vector2{1, 1.001}));
}

TEST(TestEllipseSet, ZeroRadiusInOneDimension) {
  // Test with zero radius in one dimension (line segment in 2D)
  const Vector2 center{0, 0};
  const Vector2 radii{2.0, 0.0};
  const EllipseSet ellipse{center, radii};

  // Points on the line segment
  EXPECT_TRUE(ellipse(Vector2{0, 0}));   // Center
  EXPECT_TRUE(ellipse(Vector2{1, 0}));   // On the line
  EXPECT_TRUE(ellipse(Vector2{-1, 0}));  // On the line

  // Points off the line
  EXPECT_FALSE(ellipse(Vector2{0, 0.001}));
  EXPECT_FALSE(ellipse(Vector2{1, 0.001}));
}

TEST(TestEllipseSet, Dimension) {
  const EllipseSet ellipse1d{Vector::Constant(1, 1.0), Vector::Constant(1, 1.0)};
  EXPECT_EQ(ellipse1d.dimension(), 1);

  const EllipseSet ellipse2d{Vector2{0, 0}, Vector2{1, 2}};
  EXPECT_EQ(ellipse2d.dimension(), 2);

  const EllipseSet ellipse3d{Vector3{0, 0, 0}, Vector3{1, 2, 3}};
  EXPECT_EQ(ellipse3d.dimension(), 3);

  const Vector highDim = Vector::Zero(10);
  const Vector highDimRadii = Vector::Constant(10, 1.5);
  const EllipseSet ellipseHighDim{highDim, highDimRadii};
  EXPECT_EQ(ellipseHighDim.dimension(), 10);
}

TEST(TestEllipseSet, Sampling) {
  lucid::random::seed(42);  // Set a fixed seed for reproducibility
  constexpr int n_samples = 100;
  const Vector2 center{0, 0};
  const Vector2 radii{3.0, 2.0};
  const EllipseSet ellipse{center, radii};

  // Test that all sampled points are within the ellipse
  const Matrix samples = ellipse.sample(n_samples);
  EXPECT_EQ(samples.rows(), n_samples);
  EXPECT_EQ(samples.cols(), 2);

  for (Index i = 0; i < samples.rows(); ++i) {
    const Vector2 point = samples.row(i);
    EXPECT_TRUE(ellipse(point)) << "Point (" << point(0) << ", " << point(1) << ") should be inside ellipse";
  }
}

TEST(TestEllipseSet, Sampling3D) {
  lucid::random::seed(123);
  constexpr int n_samples = 50;
  const Vector3 center{1, 2, 3};
  const Vector3 radii{2.0, 1.5, 1.0};
  const EllipseSet ellipse{center, radii};

  const Matrix samples = ellipse.sample(n_samples);
  EXPECT_EQ(samples.rows(), n_samples);
  EXPECT_EQ(samples.cols(), 3);

  for (Index i = 0; i < samples.rows(); ++i) {
    const Vector3 point = samples.row(i);
    EXPECT_TRUE(ellipse(point));
  }
}

TEST(TestEllipseSet, SamplingUniformRadius) {
  lucid::random::seed(456);
  constexpr int n_samples = 50;
  const Vector3 center{0, 0, 0};
  const Scalar radius = 2.5;
  const EllipseSet ellipse{center, radius};

  const Matrix samples = ellipse.sample(n_samples);
  EXPECT_EQ(samples.rows(), n_samples);
  EXPECT_EQ(samples.cols(), 3);

  for (Index i = 0; i < samples.rows(); ++i) {
    const Vector3 point = samples.row(i);
    EXPECT_TRUE(ellipse(point));

    // For a uniform radius, check it's a proper sphere
    const Scalar distance = (point - center).norm();
    EXPECT_LE(distance, radius + 1e-10);
  }
}

TEST(TestEllipseSet, ChangeSize) {
  Vector2 center{0, 0};
  Vector2 radii{2.0, 1.0};
  EllipseSet ellipse{center, radii};

  // Test expanding the ellipse
  ellipse.change_size(Vector2{2.0, 2.0});
  EXPECT_DOUBLE_EQ(ellipse.radii()(0), 3.0);  // 2.0 + 2.0/2
  EXPECT_DOUBLE_EQ(ellipse.radii()(1), 2.0);  // 1.0 + 2.0/2
  EXPECT_EQ(ellipse.center(), center);        // Center should not change

  // Test shrinking the ellipse
  ellipse.change_size(Vector2{-2.0, -2.0});
  EXPECT_DOUBLE_EQ(ellipse.radii()(0), 2.0);  // 3.0 - 2.0/2
  EXPECT_DOUBLE_EQ(ellipse.radii()(1), 1.0);  // 2.0 - 2.0/2
  EXPECT_EQ(ellipse.center(), center);        // Center should not change
}

TEST(TestEllipseSet, ChangeSizeUniform) {
  Vector2 center{1, 1};
  Vector2 radii{2.0, 3.0};
  EllipseSet ellipse{center, radii};

  // Test expanding uniformly using double
  ellipse.change_size(4.0);
  EXPECT_DOUBLE_EQ(ellipse.radii()(0), 4.0);  // 2.0 + 4.0/2
  EXPECT_DOUBLE_EQ(ellipse.radii()(1), 5.0);  // 3.0 + 4.0/2
  EXPECT_EQ(ellipse.center(), center);        // Center should not change
}

TEST(TestEllipseSet, ChangeSizeNonUniform) {
  Vector2 center{0, 0};
  Vector2 radii{2.0, 1.0};
  EllipseSet ellipse{center, radii};

  // Test changing size differently in each dimension
  ellipse.change_size(Vector2{2.0, 0.0});
  EXPECT_DOUBLE_EQ(ellipse.radii()(0), 3.0);  // 2.0 + 2.0/2
  EXPECT_DOUBLE_EQ(ellipse.radii()(1), 1.0);  // 1.0 + 0.0/2
}

TEST(TestEllipseSet, ChangeSizeInvalid) {
  Vector2 center{0, 0};
  Vector2 radii{1.0, 1.0};
  EllipseSet ellipse{center, radii};

  // Test that shrinking too much throws an exception
  EXPECT_THROW(ellipse.change_size(Vector2{-5.0, -5.0}), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestEllipseSet, ToRectSet) {
  const Vector2 center{1, 2};
  const Vector2 radii{3.0, 2.0};
  const EllipseSet ellipse{center, radii};

  auto rect_set = ellipse.to_rect_set();
  EXPECT_EQ(rect_set->dimension(), 2);
  EXPECT_EQ(rect_set->general_lower_bound(), center - radii);
  EXPECT_EQ(rect_set->general_upper_bound(), center + radii);
}

TEST(TestEllipseSet, Equality) {
  const Vector2 center1{1, 2};
  const Vector2 radii1{3.0, 2.0};
  const EllipseSet ellipse1{center1, radii1};
  const EllipseSet ellipse2{center1, radii1};
  const EllipseSet ellipse3{Vector2{1, 2}, Vector2{3.0, 2.1}};

  EXPECT_TRUE(ellipse1 == ellipse2);
  EXPECT_FALSE(ellipse1 == ellipse3);
}

TEST(TestEllipseSet, EqualityWithSet) {
  const Vector2 center{1, 2};
  const Vector2 radii{3.0, 2.0};
  const EllipseSet ellipse1{center, radii};
  const EllipseSet ellipse2{center, radii};

  const Set& set1 = ellipse1;
  const Set& set2 = ellipse2;

  EXPECT_TRUE(set1 == set2);
}

TEST(TestEllipseSet, Lattice2D) {
  const Vector2 center{0, 0};
  const Vector2 radii{2.0, 1.0};
  const EllipseSet ellipse{center, radii};

  const VectorI points_per_dim{5, 5};
  const Matrix lattice = ellipse.lattice(points_per_dim, false);

  // All lattice points should be inside the ellipse
  for (Index i = 0; i < lattice.rows(); ++i) {
    EXPECT_TRUE(ellipse(lattice.row(i)));
  }

  // The number of points should be less than the full grid (since it's filtered)
  EXPECT_LT(lattice.rows(), 25);  // Less than 5*5
  EXPECT_GT(lattice.rows(), 0);   // But more than 0
}

TEST(TestEllipseSet, Lattice3D) {
  const Vector3 center{0, 0, 0};
  const Vector3 radii{1.0, 1.0, 1.0};
  const EllipseSet ellipse{center, radii};

  const VectorI points_per_dim{{3, 3, 3}};
  const Matrix lattice = ellipse.lattice(points_per_dim, true);

  // All lattice points should be inside the ellipse
  for (Index i = 0; i < lattice.rows(); ++i) {
    EXPECT_TRUE(ellipse(lattice.row(i)));
  }

  EXPECT_GT(lattice.rows(), 0);
}

TEST(TestEllipseSet, InvalidConstruction) {
  // Test with mismatched dimensions
  const Vector2 center{0, 0};
  const Vector3 radii{1, 1, 1};
  EXPECT_THROW(EllipseSet(center, radii), lucid::exception::LucidInvalidArgumentException);

  // Test with negative radius
  const Vector2 negative_radii{1.0, -1.0};
  EXPECT_THROW(EllipseSet(center, negative_radii), lucid::exception::LucidInvalidArgumentException);

  // Test with empty center
  const Vector empty_center{};
  const Vector empty_radii{};
  EXPECT_THROW(EllipseSet(empty_center, empty_radii), lucid::exception::LucidInvalidArgumentException);
}
