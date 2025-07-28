/**
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/SphereSet.h"
#include "lucid/util/exception.h"
#include "lucid/util/random.h"

using lucid::Index;
using lucid::Matrix;
using lucid::Scalar;
using lucid::Set;
using lucid::SphereSet;
using lucid::Vector;
using lucid::Vector2;
using lucid::Vector3;
using lucid::VectorI;

// Test basic construction and contains functionality
TEST(TestSphereSet, Construction) {
  const Vector2 center{1.5, -2.5};
  const Scalar radius = 3.0;
  const SphereSet sphere{center, radius};

  EXPECT_EQ(sphere.dimension(), 2);
  EXPECT_EQ(sphere.center(), center);
  EXPECT_EQ(sphere.radius(), radius);
}

TEST(TestSphereSet, Contains2D) {
  const Vector2 center{0, 0};
  const Scalar radius = 1.0;
  const SphereSet sphere{center, radius};

  // Points inside the sphere
  EXPECT_TRUE(sphere(Vector2{0, 0}));      // Center
  EXPECT_TRUE(sphere(Vector2{0.5, 0.5}));  // Inside
  EXPECT_TRUE(sphere(Vector2{0.7, 0.7}));  // Close to boundary but inside

  // Points on the boundary
  EXPECT_TRUE(sphere(Vector2{1, 0}));                                    // Boundary
  EXPECT_TRUE(sphere(Vector2{0, 1}));                                    // Boundary
  EXPECT_TRUE(sphere(Vector2{-1, 0}));                                   // Boundary
  EXPECT_TRUE(sphere(Vector2{0, -1}));                                   // Boundary
  EXPECT_TRUE(sphere(Vector2{sqrt(2) / 2 - 0.01, sqrt(2) / 2 - 0.01}));  // sqrt(2)/2, on boundary

  // Points outside the sphere
  EXPECT_FALSE(sphere(Vector2{1.1, 0}));    // Outside
  EXPECT_FALSE(sphere(Vector2{0, 1.1}));    // Outside
  EXPECT_FALSE(sphere(Vector2{1.5, 1.5}));  // Far outside
  EXPECT_FALSE(sphere(Vector2{-2, -2}));    // Far outside
}

TEST(TestSphereSet, Contains3D) {
  const Vector3 center{1, 2, 3};
  const Scalar radius = 2.0;
  const SphereSet sphere{center, radius};

  // Points inside the sphere
  EXPECT_TRUE(sphere(Vector3{1, 2, 3}));  // Center
  EXPECT_TRUE(sphere(Vector3{2, 2, 3}));  // Inside (distance = 1)
  EXPECT_TRUE(sphere(Vector3{1, 3, 3}));  // Inside (distance = 1)
  EXPECT_TRUE(sphere(Vector3{1, 2, 4}));  // Inside (distance = 1)

  // Points on the boundary
  EXPECT_TRUE(sphere(Vector3{3, 2, 3}));   // Boundary (distance = 2)
  EXPECT_TRUE(sphere(Vector3{1, 4, 3}));   // Boundary (distance = 2)
  EXPECT_TRUE(sphere(Vector3{1, 2, 5}));   // Boundary (distance = 2)
  EXPECT_TRUE(sphere(Vector3{-1, 2, 3}));  // Boundary (distance = 2)

  // Points outside the sphere
  EXPECT_FALSE(sphere(Vector3{4, 2, 3}));  // Outside (distance = 3)
  EXPECT_FALSE(sphere(Vector3{1, 5, 3}));  // Outside (distance = 3)
  EXPECT_FALSE(sphere(Vector3{5, 5, 5}));  // Far outside
}

TEST(TestSphereSet, ContainsOffCenter) {
  const Vector2 center{-2, 3};
  const Scalar radius = 1.5;
  const SphereSet sphere{center, radius};

  // Points inside the sphere
  EXPECT_TRUE(sphere(Vector2{-2, 3}));    // Center
  EXPECT_TRUE(sphere(Vector2{-1.5, 3}));  // Inside
  EXPECT_TRUE(sphere(Vector2{-2, 2}));    // Inside

  // Points on the boundary
  EXPECT_TRUE(sphere(Vector2{-0.5, 3}));  // Boundary (distance = 1.5)
  EXPECT_TRUE(sphere(Vector2{-2, 4.5}));  // Boundary (distance = 1.5)

  // Points outside the sphere
  EXPECT_FALSE(sphere(Vector2{0, 3}));   // Outside (distance = 2)
  EXPECT_FALSE(sphere(Vector2{-2, 5}));  // Outside (distance = 2)
}

// Test zero radius (point set)
TEST(TestSphereSet, ZeroRadius) {
  const Vector2 center{1, 1};
  const Scalar radius = 0.0;
  const SphereSet sphere{center, radius};

  // Only the center point should be contained
  EXPECT_TRUE(sphere(Vector2{1, 1}));

  // Any other point should not be contained
  EXPECT_FALSE(sphere(Vector2{1.001, 1}));
  EXPECT_FALSE(sphere(Vector2{1, 1.001}));
  EXPECT_FALSE(sphere(Vector2{0.999, 1}));
  EXPECT_FALSE(sphere(Vector2{1, 0.999}));
}

// Test dimension consistency
TEST(TestSphereSet, Dimension) {
  const SphereSet sphere1d{Vector::Constant(1, 1.0), 1.0};
  EXPECT_EQ(sphere1d.dimension(), 1);

  const SphereSet sphere2d{Vector2{0, 0}, 1.0};
  EXPECT_EQ(sphere2d.dimension(), 2);

  const SphereSet sphere3d{Vector3{0, 0, 0}, 1.0};
  EXPECT_EQ(sphere3d.dimension(), 3);

  const Vector highDim = Vector::Zero(10);
  const SphereSet sphereHighDim{highDim, 1.0};
  EXPECT_EQ(sphereHighDim.dimension(), 10);
}

// Test sampling functionality
TEST(TestSphereSet, Sampling) {
  lucid::random::seed(42);  // Set a fixed seed for reproducibility
  constexpr int n_samples = 5;
  const Vector2 center{0, 0};
  constexpr Scalar radius = 2.0;
  const SphereSet sphere{center, radius};

  // Test that all sampled points are within the sphere
  const Matrix samples = sphere.sample(n_samples);
  EXPECT_EQ(samples.rows(), n_samples);
  EXPECT_EQ(samples.cols(), 2);

  for (Index i = 0; i < samples.rows(); ++i) {
    const Vector2 point = samples.row(i);
    EXPECT_TRUE(sphere(point));

    // Check that the distance from center is within radius
    const Scalar distance = (point - center).norm();
    EXPECT_LE(distance, radius + 1e-10);
  }
}

// Test sampling distribution properties
TEST(TestSphereSet, SamplingDistribution) {
  lucid::random::seed(42);  // Set a fixed seed for reproducibility
  const Vector2 center{0, 0};
  const Scalar radius = 5.0;
  const SphereSet sphere{center, radius};

  const Matrix samples = sphere.sample(1000);

  // Check basic properties
  EXPECT_EQ(samples.rows(), 1000);
  EXPECT_EQ(samples.cols(), 2);

  // Statistical tests (with some tolerance for randomness)
  double mean_distance = 0.0;
  int points_in_inner_half = 0;

  for (Index i = 0; i < samples.rows(); ++i) {
    const Vector2 point = samples.row(i);
    const Scalar distance = (point - center).norm();

    EXPECT_LE(distance, radius + 1e-10);
    mean_distance += distance;

    if (distance <= radius / 2.0) {
      points_in_inner_half++;
    }
  }

  mean_distance /= samples.rows();

  // For a uniform distribution in a sphere, the mean distance should be around 2/3 * radius
  // We allow some tolerance for the random sampling
  EXPECT_GT(mean_distance, radius * 0.5);
  EXPECT_LT(mean_distance, radius * 0.8);

  // The inner half-radius disk should contain approximately 25% of points in 2D
  // (area ratio = (r/2)^2 / r^2 = 1/4)
  const double inner_ratio = static_cast<double>(points_in_inner_half) / samples.rows();
  EXPECT_GT(inner_ratio, 0.15);  // Allow some tolerance
  EXPECT_LT(inner_ratio, 0.35);
}

TEST(TestSphereSet, SamplingOffCenter) {
  lucid::random::seed(42);  // Set a fixed seed for reproducibility
  const Vector3 center{1, -2, 3};
  const Scalar radius = 1.5;
  const SphereSet sphere{center, radius};

  const Matrix samples = sphere.sample(50);
  EXPECT_EQ(samples.rows(), 50);
  EXPECT_EQ(samples.cols(), 3);

  for (Index i = 0; i < samples.rows(); ++i) {
    const Vector3 point = samples.row(i);
    EXPECT_TRUE(sphere(point));

    const Scalar distance = (point - center).norm();
    EXPECT_LE(distance, radius + 1e-10);
  }
}

TEST(TestSphereSet, SamplingZeroRadius) {
  lucid::random::seed(42);  // Set a fixed seed for reproducibility
  const Vector2 center{5, -3};
  const Scalar radius = 0.0;
  const SphereSet sphere{center, radius};

  const Matrix samples = sphere.sample(10);
  EXPECT_EQ(samples.rows(), 10);
  EXPECT_EQ(samples.cols(), 2);

  // All samples should be exactly at the center
  for (Index i = 0; i < samples.rows(); ++i) {
    const Vector2 point = samples.row(i);
    EXPECT_TRUE(sphere(point));
    EXPECT_NEAR((point - center).norm(), 0.0, 1e-10);
  }
}

// Test exception handling
TEST(TestSphereSet, InvalidArguments) {
  // Test empty center vector
  EXPECT_THROW(SphereSet(Vector{}, 1.0), lucid::exception::LucidInvalidArgumentException);

  // Test negative radius
  EXPECT_THROW(SphereSet(Vector2{0, 0}, -1.0), lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(SphereSet(Vector2{0, 0}, -0.1), lucid::exception::LucidInvalidArgumentException);
}

// Test contains with wrong dimensions
TEST(TestSphereSet, WrongDimensionContains) {
  const SphereSet sphere2d{Vector2{0, 0}, 1.0};
  const SphereSet sphere3d{Vector3{0, 0, 0}, 1.0};

  // Test checking points with wrong dimensions
  EXPECT_THROW(static_cast<void>(sphere2d(Vector3{0, 0, 0})), lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(static_cast<void>(sphere3d(Vector2{0, 0})), lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(static_cast<void>(sphere2d(Vector::Constant(0, 0.0))), lucid::exception::LucidInvalidArgumentException);
}

// Test mathematical properties
TEST(TestSphereSet, MathematicalProperties) {
  const Vector3 center{2, -1, 3};
  const Scalar radius = 4.0;
  const SphereSet sphere{center, radius};

  // Test that distance function works correctly
  const Vector3 point_on_boundary = center + Vector3{radius, 0, 0};
  EXPECT_TRUE(sphere(point_on_boundary));
  EXPECT_NEAR((point_on_boundary - center).norm(), radius, 1e-10);

  // Test multiple points on the boundary
  const Vector3 point_on_boundary2 = center + Vector3{0, radius, 0};
  const Vector3 point_on_boundary3 = center + Vector3{0, 0, radius};
  const Vector3 point_on_boundary4 = center + Vector3{-radius, 0, 0};

  EXPECT_TRUE(sphere(point_on_boundary2));
  EXPECT_TRUE(sphere(point_on_boundary3));
  EXPECT_TRUE(sphere(point_on_boundary4));

  // Test points just outside the boundary
  const Vector3 point_outside = center + Vector3{radius + 1e-6, 0, 0};
  EXPECT_FALSE(sphere(point_outside));

  // Test diagonal points on boundary
  const Scalar coord = radius / std::sqrt(3.0);  // radius / sqrt(3) for 3D unit vector
  const Vector3 diagonal_point = center + Vector3{coord, coord, coord};
  EXPECT_TRUE(sphere(diagonal_point));
  EXPECT_NEAR((diagonal_point - center).norm(), radius, 1e-10);
}

// Test edge cases with very small and very large values
TEST(TestSphereSet, EdgeCases) {
  // Very small radius
  const SphereSet tinyRad{Vector2{0, 0}, 1e-8};
  EXPECT_TRUE(tinyRad(Vector2{0, 0}));
  EXPECT_FALSE(tinyRad(Vector2{1e-7, 0}));

  // Very large radius
  const SphereSet largeRad{Vector2{0, 0}, 1e6};
  EXPECT_TRUE(largeRad(Vector2{0, 0}));
  EXPECT_TRUE(largeRad(Vector2{1e5, 1e5}));
  EXPECT_FALSE(largeRad(Vector2{1e6, 1e6}));  // sqrt(2) * 1e6 > 1e6

  // Very large center coordinates
  const SphereSet largeCenter{Vector2{1e6, -1e6}, 1.0};
  EXPECT_TRUE(largeCenter(Vector2{1e6, -1e6}));
  EXPECT_TRUE(largeCenter(Vector2{1e6 + 0.5, -1e6}));
  EXPECT_FALSE(largeCenter(Vector2{1e6 + 2, -1e6}));
}

// Test precision around boundary
TEST(TestSphereSet, BoundaryPrecision) {
  const Vector2 center{0, 0};
  const Scalar radius = 1.0;
  const SphereSet sphere{center, radius};

  // Test points very close to the boundary
  const Scalar eps = 1e-14;
  EXPECT_TRUE(sphere(Vector2{1.0 - eps, 0}));   // Just inside
  EXPECT_FALSE(sphere(Vector2{1.0 + eps, 0}));  // Just outside

  // Test at exact boundary with floating point precision
  const Scalar coord = std::sqrt(2.0) / 2.0;
  EXPECT_TRUE(sphere(Vector2{coord, coord}));
  EXPECT_TRUE(sphere(Vector2{-coord, coord}));
  EXPECT_TRUE(sphere(Vector2{coord, -coord}));
  EXPECT_TRUE(sphere(Vector2{-coord, -coord}));
}

// Test with 1D sphere (circle endpoints)
TEST(TestSphereSet, OneDimensional) {
  lucid::random::seed(42);  // Set a fixed seed for reproducibility
  const Vector center{Vector::Constant(1, 0.0)};
  const Scalar radius = 2.0;
  const SphereSet sphere1d{center, radius};

  EXPECT_EQ(sphere1d.dimension(), 1);

  // Test contains for 1D case
  EXPECT_TRUE(sphere1d(Vector::Constant(1, 0.0)));    // Center
  EXPECT_TRUE(sphere1d(Vector::Constant(1, 1.0)));    // Inside
  EXPECT_TRUE(sphere1d(Vector::Constant(1, -1.5)));   // Inside
  EXPECT_TRUE(sphere1d(Vector::Constant(1, 2.0)));    // Boundary
  EXPECT_TRUE(sphere1d(Vector::Constant(1, -2.0)));   // Boundary
  EXPECT_FALSE(sphere1d(Vector::Constant(1, 2.1)));   // Outside
  EXPECT_FALSE(sphere1d(Vector::Constant(1, -2.1)));  // Outside

  // Test sampling for 1D case
  const Matrix samples1d = sphere1d.sample(50);
  EXPECT_EQ(samples1d.rows(), 50);
  EXPECT_EQ(samples1d.cols(), 1);

  for (Index i = 0; i < samples1d.rows(); ++i) {
    const Vector point = samples1d.row(i);
    EXPECT_TRUE(sphere1d(point));
    EXPECT_LE(std::abs(point(0) - center(0)), radius + 1e-10);
  }
}

TEST(TestSphereSet, Lattice2DInclude) {
  const Vector2 center{1, 1};
  constexpr Scalar radius = 2.0;

  const SphereSet set{center, radius};

  const Matrix lattice{set.lattice(VectorI::Constant(2, 5), true)};
  EXPECT_LE(lattice.rows(), 5 * 5);  // 5 points per dimension
  EXPECT_EQ(lattice.cols(), 2);
  for (Index i = 0; i < lattice.rows(); ++i) ASSERT_TRUE(set(lattice.row(i)));
}

TEST(TestSphereSet, Lattice2D) {
  const Vector2 center{1, 1};
  constexpr Scalar radius = 2.0;

  const SphereSet set{center, radius};

  const Matrix lattice{set.lattice(VectorI::Constant(2, 5), false)};
  EXPECT_LE(lattice.rows(), 5 * 5);  // 5 points per dimension
  EXPECT_EQ(lattice.cols(), 2);
  for (Index i = 0; i < lattice.rows(); ++i) ASSERT_TRUE(set(lattice.row(i)));
}

TEST(TestSphereSet, Lattice3D) {
  const Vector3 center{1, 1, 4};
  constexpr Scalar radius = 3.0;

  const SphereSet set{center, radius};

  const Matrix lattice{set.lattice(VectorI::Constant(3, 10), false)};
  EXPECT_LE(lattice.rows(), 10 * 10 * 10);  // 5 points per dimension
  EXPECT_EQ(lattice.cols(), 3);
  for (Index i = 0; i < lattice.rows(); ++i) ASSERT_TRUE(set(lattice.row(i)));
}

// Test polymorphic behavior (inheritance from Set)
TEST(TestSphereSet, PolymorphicBehavior) {
  lucid::random::seed(42);  // Set a fixed seed for reproducibility
  const Vector2 center{1, 1};
  constexpr Scalar radius = 2.0;

  const std::unique_ptr<Set> set{std::make_unique<SphereSet>(center, radius)};

  // Test virtual methods work correctly
  EXPECT_EQ(set->dimension(), 2);
  EXPECT_TRUE((*set)(center));              // operator() should work
  EXPECT_TRUE((*set)(Vector2{2.0, 1.0}));   // Point inside
  EXPECT_FALSE((*set)(Vector2{4.0, 1.0}));  // Point outside

  // Test sampling through base class interface
  const Matrix samples = set->sample(10);
  EXPECT_EQ(samples.rows(), 10);
  EXPECT_EQ(samples.cols(), 2);

  for (Index i = 0; i < samples.rows(); ++i) {
    EXPECT_TRUE((*set)(samples.row(i).transpose()));
  }

  const VectorI points_per_dim = VectorI::Constant(2, 5);
  const Matrix lattice{set->lattice(points_per_dim, false)};
  for (Index i = 0; i < lattice.rows(); ++i) {
    ASSERT_TRUE((*set)(lattice.row(i)));
  }
}

// Test high-dimensional sphere
TEST(TestSphereSet, HighDimensional) {
  lucid::random::seed(42);  // Set a fixed seed for reproducibility
  const int dim = 7;
  const Vector center = Vector::Ones(dim) * 2.0;  // Center at (2,2,2,2,2,2,2)
  const Scalar radius = 3.0;
  const SphereSet sphereHigh{center, radius};

  EXPECT_EQ(sphereHigh.dimension(), dim);

  // Test center
  EXPECT_TRUE(sphereHigh(center));

  // Test point at distance 1 from center in first dimension
  Vector testPoint = center;
  testPoint(0) += 1.0;
  EXPECT_TRUE(sphereHigh(testPoint));

  // Test point at exact radius distance
  testPoint = center;
  testPoint(0) += radius;
  EXPECT_TRUE(sphereHigh(testPoint));

  // Test point outside
  testPoint = center;
  testPoint(0) += radius + 0.1;
  EXPECT_FALSE(sphereHigh(testPoint));

  // Test sampling
  const Matrix samplesHigh = sphereHigh.sample(20);
  EXPECT_EQ(samplesHigh.rows(), 20);
  EXPECT_EQ(samplesHigh.cols(), dim);

  for (Index i = 0; i < samplesHigh.rows(); ++i) {
    const Vector point = samplesHigh.row(i);
    EXPECT_TRUE(sphereHigh(point));
    const Scalar distance = (point - center).norm();
    EXPECT_LE(distance, radius + 1e-10);
  }
}
