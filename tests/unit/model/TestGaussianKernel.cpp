/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "lucid/model/GaussianKernel.h"

using lucid::ConstMatrixRef;
using lucid::GaussianKernel;
using lucid::Index;
using lucid::Matrix;
using lucid::Parameter;
using lucid::Vector;

namespace {

/**
 * Compute the Gaussian kernel between two vectors.
 * The Gaussian kernel is defined as:
 * @f[
 * k(x_1, x_2) = \sigma_f^2 \exp\left(-\frac{1}{2} (x_1 - x_2)^T \Sigma (x_1 - x_2)\right)
 * @f]
 * where @f$ \Sigma = \text{diag}(\sigma_l)^{-2} @f$.
 * @param x1 @f$ x_1 @f$ vector
 * @param x2 @f$ x_2 @f$ vector
 * @param sigma_l @f$ \sigma_l @f$ vector
 * @param sigma_f @f$ \sigma_f @f$ value
 * @return Gaussian kernel value
 */
inline Matrix gaussian(ConstMatrixRef x1, ConstMatrixRef x2, const Vector& sigma_l, const double sigma_f) {
  const Matrix def_num = Matrix::NullaryExpr(x1.rows(), x2.rows(), [&](const Index row, const Index col) {
    return (x1.row(row) - x2.row(col)) * sigma_l.cwiseProduct(sigma_l).cwiseInverse().asDiagonal() *
           (x1.row(row) - x2.row(col)).transpose();
  });
  return sigma_f * sigma_f * (-0.5 * def_num.array()).exp();
}

}  // namespace

class TestGaussianKernel : public ::testing::Test {
 protected:
  static Vector get_sigma_l() {
    Vector sigma_l{4};
    // sigma_l << 3.2, 5.1, 3.4, 1.24;
    sigma_l << 2.1, 2.1, 2.1, 2.1;  // For simplicity, use the same value for all dimensions
    return sigma_l;
  }

  TestGaussianKernel() : sigma_f_{4.2}, sigma_l_{get_sigma_l()}, kernel_{get_sigma_l(), sigma_f_} {}

  double sigma_f_;
  Vector sigma_l_;
  GaussianKernel kernel_;
};

TEST_F(TestGaussianKernel, VectorConstructor) {
  const GaussianKernel kernel{Vector::Constant(3, 1.0)};
  EXPECT_EQ(kernel.sigma_f(), 1.0);
  EXPECT_EQ(kernel.sigma_l().size(), 3);
  EXPECT_THAT(kernel.sigma_l(), ::testing::Each(1.0));
  // EXPECT_EQ(kernel.sigma_f(), kernel.get<double>(Parameter::MEAN));
  EXPECT_EQ(kernel.sigma_f(), kernel.get<double>(Parameter::SIGMA_F));
  // EXPECT_EQ(kernel.sigma_l(), kernel.get<const Vector&>(Parameter::LENGTH_SCALE));
  // EXPECT_EQ(kernel.sigma_l(), kernel.get<const Vector&>(Parameter::COVARIANCE));
  EXPECT_EQ(kernel.sigma_l(), kernel.get<const Vector&>(Parameter::SIGMA_L));
}

TEST_F(TestGaussianKernel, DimensionConstructor) {
  const GaussianKernel kernel{4, 3.0};
  EXPECT_EQ(kernel.sigma_f(), 1.0);
  EXPECT_EQ(kernel.sigma_l().size(), 4);
  EXPECT_THAT(kernel.sigma_l(), ::testing::Each(3.0));
  // EXPECT_EQ(kernel.sigma_f(), kernel.get<double>(Parameter::MEAN));
  EXPECT_EQ(kernel.sigma_f(), kernel.get<double>(Parameter::SIGMA_F));
  // EXPECT_EQ(kernel.sigma_l(), kernel.get<const Vector&>(Parameter::LENGTH_SCALE));
  // EXPECT_EQ(kernel.sigma_l(), kernel.get<const Vector&>(Parameter::COVARIANCE));
  EXPECT_EQ(kernel.sigma_l(), kernel.get<const Vector&>(Parameter::SIGMA_L));
}

TEST_F(TestGaussianKernel, VectorCorrectnessIsotropic) {
  const Vector sigma_l{Vector::Constant(3, 4.5)};  // Use the same value for all dimensions
  const Vector x1{Vector::Random(3)}, x2{Vector::Random(3)};

  // ||(x1 / σl) - (x2 / σl)||^2
  const double num = (x1.array() / sigma_l.array() - x2.array() / sigma_l.array()).matrix().squaredNorm();
  // σf^2 * exp(-0.5 * ||(x1 / σl) - (x2 / σl)||^2)
  const double computed = sigma_f_ * sigma_f_ * std::exp(-0.5 * num);

  ASSERT_EQ(gaussian(x1, x2, sigma_l, sigma_f_).size(), 1);
  EXPECT_DOUBLE_EQ(computed, gaussian(x1, x2, sigma_l, sigma_f_).value());
}

TEST_F(TestGaussianKernel, VectorCorrectnessAnisotropic) {
  const Vector sigma_l{Vector::LinSpaced(3, 3, 4.5)};  // Use the same value for all dimensions
  const Vector x1{Vector::Random(3)}, x2{Vector::Random(3)};

  // ||(x1 / σl) - (x2 / σl)||^2
  const double num = (x1.array() / sigma_l.array() - x2.array() / sigma_l.array()).matrix().squaredNorm();
  // σf^2 * exp(-0.5 * ||(x1 / σl) - (x2 / σl)||^2)
  const double computed = sigma_f_ * sigma_f_ * std::exp(-0.5 * num);

  ASSERT_EQ(gaussian(x1, x2, sigma_l, sigma_f_).size(), 1);
  EXPECT_DOUBLE_EQ(computed, gaussian(x1, x2, sigma_l, sigma_f_).value());
}

TEST_F(TestGaussianKernel, MatrixCorrectnessIsotropic) {
  const Vector sigma_l{Vector::Constant(3, 4.5)};  // Use the same value for all dimensions
  const Matrix x1{Matrix::Random(2, 3)}, x2{Matrix::Random(4, 3)};

  // ||(x1 / σl) - (x2 / σl)||^2
  const Matrix num = Matrix::NullaryExpr(x1.rows(), x2.rows(), [&](const Index row, const Index col) {
    return (x1.row(row).array() / sigma_l.array() - x2.row(col).array() / sigma_l.array()).matrix().squaredNorm();
  });
  // σf^2 * exp(-0.5 * ||(x1 / σl) - (x2 / σl)||^2)
  const Matrix computed = sigma_f_ * sigma_f_ * (-0.5 * num.array()).exp();

  EXPECT_TRUE(computed.isApprox(gaussian(x1, x2, sigma_l, sigma_f_)));
}

TEST_F(TestGaussianKernel, MatrixCorrectnessAnisotropic) {
  const Vector sigma_l{Vector::LinSpaced(3, 3, 4.5)};  // Use the same value for all dimensions
  const Matrix x1{Matrix::Random(3, 3)}, x2{Matrix::Random(2, 3)};

  // ||(x1 / σl) - (x2 / σl)||^2
  const Matrix num = Matrix::NullaryExpr(x1.rows(), x2.rows(), [&](const Index row, const Index col) {
    return (x1.row(row).array() / sigma_l.array() - x2.row(col).array() / sigma_l.array()).matrix().squaredNorm();
  });
  // σf^2 * exp(-0.5 * ||(x1 / σl) - (x2 / σl)||^2)
  const Matrix computed = sigma_f_ * sigma_f_ * (-0.5 * num.array()).exp();

  EXPECT_TRUE(computed.isApprox(gaussian(x1, x2, sigma_l, sigma_f_)));
}

TEST_F(TestGaussianKernel, ApplyVectorIsotropic) {
  const Vector sigma_l{Vector::Constant(4, 2.1)};
  const Vector x1{Vector::Random(4)}, x2{Vector::Random(4)};
  const GaussianKernel kernel{sigma_l, sigma_f_};
  EXPECT_DOUBLE_EQ(kernel(x1, x2).value(), gaussian(x1, x2, sigma_l, sigma_f_).value());
}
TEST_F(TestGaussianKernel, ApplyVectorAnisotropic) {
  const Vector sigma_l{Vector::LinSpaced(4, 1, 2.1)};
  const Vector x1{Vector::Random(4)}, x2{Vector::Random(4)};
  const GaussianKernel kernel{sigma_l, sigma_f_};
  EXPECT_DOUBLE_EQ(kernel(x1, x2).value(), gaussian(x1, x2, sigma_l, sigma_f_).value());
}

TEST_F(TestGaussianKernel, ApplyMatrixIsotropic) {
  const Vector sigma_l{Vector::Constant(4, 2.1)};
  const Matrix x1{Matrix::Random(4, 4)}, x2{Matrix::Random(3, 4)};
  const GaussianKernel kernel{sigma_l, sigma_f_};
  EXPECT_TRUE(kernel(x1, x2).isApprox(gaussian(x1, x2, sigma_l, sigma_f_)));
}
TEST_F(TestGaussianKernel, ApplyMatrixAnisotropic) {
  const Vector sigma_l{Vector::LinSpaced(4, 1, 2.1)};
  const Matrix x1{Matrix::Random(4, 4)}, x2{Matrix::Random(3, 4)};
  const GaussianKernel kernel{sigma_l, sigma_f_};
  EXPECT_TRUE(kernel(x1, x2).isApprox(gaussian(x1, x2, sigma_l, sigma_f_)));
}
