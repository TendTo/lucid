/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "lucid/model/ValleePoussinKernel.h"

using lucid::ConstMatrixRef;
using lucid::Index;
using lucid::Matrix;
using lucid::Parameter;
using lucid::ValleePoussinKernel;
using lucid::Vector;

namespace {

/**
 * Compute the Vallee-Poussin kernel for a vector.
 * The Vallee-Poussin kernel is defined as:
 * @f[
 * k_{a,b}^n(x) = \frac{1}{(b - a)^n} \prod_{i=1}^{n} \frac{
 * \sin\left( \frac{b + a}{2} x_i \right)
 * \sin\left( \frac{b - a}{2} x_i \right)
 * }{
 * \sin^2{\left( \frac{x_i}{2} \right)}
 * }
 * @f]
 * @param x input matrix (each row is a sample)
 * @param a parameter a
 * @param b parameter b
 * @return Vallee-Poussin kernel values
 */
Vector vallee_poussin(ConstMatrixRef x, const double a, const double b) {
  const double coeff = 1.0 / std::pow(b - a, static_cast<double>(x.cols()));
  Vector prod = Vector::Ones(x.rows());

  for (Index i = 0; i < x.cols(); ++i) {
    const auto& col = x.col(i);
    const auto numerator = (((b + a) / 2.0 * col.array()).sin() * ((b - a) / 2.0 * col.array()).sin()).matrix();
    const auto denominator = ((col.array() / 2.0).sin().square()).matrix();

    Vector fraction{x.rows()};
    for (Index j = 0; j < x.rows(); ++j) {
      if (denominator(j) != 0.0) {
        fraction(j) = numerator(j) / denominator(j);
      } else {
        fraction(j) = b * b - a * a;
      }
    }
    prod.array() *= fraction.array();
  }

  return coeff * prod;
}

}  // namespace

class TestValleePoussinKernel : public ::testing::Test {
 protected:
  TestValleePoussinKernel() : a_{1.5}, b_{3.0}, kernel_{a_, b_} {}

  double a_;
  double b_;
  ValleePoussinKernel kernel_;
};

TEST_F(TestValleePoussinKernel, DefaultConstructor) {
  const ValleePoussinKernel kernel{};
  EXPECT_EQ(kernel.a(), 1.0);
  EXPECT_EQ(kernel.b(), 1.0);
}

TEST_F(TestValleePoussinKernel, ParameterizedConstructor) {
  const ValleePoussinKernel kernel{2.5, 4.5};
  EXPECT_EQ(kernel.a(), 2.5);
  EXPECT_EQ(kernel.b(), 4.5);
}

TEST_F(TestValleePoussinKernel, IsStationary) { EXPECT_TRUE(kernel_.is_stationary()); }

TEST_F(TestValleePoussinKernel, ParametersList) {
  EXPECT_THAT(kernel_.parameters_list(), ::testing::UnorderedElementsAre(Parameter::A, Parameter::B));
}

TEST_F(TestValleePoussinKernel, ParametersHas) {
  EXPECT_TRUE(kernel_.has(Parameter::A));
  EXPECT_TRUE(kernel_.has(Parameter::B));
  EXPECT_FALSE(kernel_.has(Parameter::SIGMA_L));
  EXPECT_FALSE(kernel_.has(Parameter::SIGMA_F));
}

TEST_F(TestValleePoussinKernel, ParametersGet) {
  EXPECT_EQ(kernel_.get<Parameter::A>(), a_);
  EXPECT_EQ(kernel_.get<Parameter::B>(), b_);
}

TEST_F(TestValleePoussinKernel, ParametersSet) {
  constexpr double new_a = 2.0;
  constexpr double new_b = 5.0;

  EXPECT_NO_THROW(kernel_.set(Parameter::A, new_a));
  EXPECT_NO_THROW(kernel_.set(Parameter::B, new_b));

  EXPECT_EQ(kernel_.get<Parameter::A>(), new_a);
  EXPECT_EQ(kernel_.get<Parameter::B>(), new_b);
  EXPECT_EQ(kernel_.a(), new_a);
  EXPECT_EQ(kernel_.b(), new_b);
}

TEST_F(TestValleePoussinKernel, Clone) {
  const auto cloned = kernel_.clone();
  ASSERT_NE(cloned, nullptr);

  auto* vp_kernel = dynamic_cast<ValleePoussinKernel*>(cloned.get());
  ASSERT_NE(vp_kernel, nullptr);
  EXPECT_EQ(vp_kernel->a(), a_);
  EXPECT_EQ(vp_kernel->b(), b_);
}

TEST_F(TestValleePoussinKernel, VectorCorrectness) {
  const Matrix x{Matrix::Random(5, 3)};
  const Vector expected = vallee_poussin(x, a_, b_);
  const Vector result = kernel_(x);

  ASSERT_EQ(result.size(), x.rows());
  EXPECT_TRUE(result.isApprox(expected));
}

TEST_F(TestValleePoussinKernel, ZeroHandling) {
  const ValleePoussinKernel kernel{1.0, 2.0};
  Matrix x{2, 2};
  x << 0.0, 0.0, 1.0, 2.0;

  const Vector result = kernel(x);

  ASSERT_EQ(result.size(), 2);
  // When x_i = 0, the fraction should be b^2 - a^2
  EXPECT_FALSE(result.hasNaN());
  EXPECT_FALSE(std::isinf(result(0)));
}

TEST_F(TestValleePoussinKernel, LargeInputDimensions) {
  const Matrix x{Matrix::Random(10, 5)};
  const Vector result = kernel_(x);

  ASSERT_EQ(result.size(), 10);
  EXPECT_FALSE(result.hasNaN());
  EXPECT_TRUE(result.allFinite());
}

TEST_F(TestValleePoussinKernel, ConsistencyAcrossMultipleCalls) {
  const Matrix x{Matrix::Random(4, 3)};

  const Vector result1 = kernel_(x);
  const Vector result2 = kernel_(x);
  const Vector result3 = kernel_(x);

  EXPECT_TRUE(result1.isApprox(result2));
  EXPECT_TRUE(result2.isApprox(result3));
}
