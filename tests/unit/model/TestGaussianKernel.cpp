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
inline Matrix gaussian(ConstMatrixRef x1, ConstMatrixRef x2, const Vector& sigma_l, const double sigma_f = 1.0) {
  const Matrix def_num = Matrix::NullaryExpr(x1.rows(), x2.rows(), [&](const Index row, const Index col) {
    return (x1.row(row) - x2.row(col)) * sigma_l.cwiseProduct(sigma_l).cwiseInverse().asDiagonal() *
           (x1.row(row) - x2.row(col)).transpose();
  });
  return sigma_f * sigma_f * (-0.5 * def_num.array()).exp();
}
/** @overload  */
inline Matrix gaussian(ConstMatrixRef x1, ConstMatrixRef x2, const double sigma_l, const double sigma_f = 1.0) {
  return gaussian(x1, x2, Vector::Constant(x1.cols(), sigma_l), sigma_f);
}

}  // namespace

class TestGaussianKernel : public ::testing::Test {
 protected:
  TestGaussianKernel()
      : sigma_f_{4.2}, is_sigma_l_{2.4}, an_sigma_l_{Vector::LinSpaced(4, 1.2, 6.3)}, kernel_{an_sigma_l_, sigma_f_} {}

  double sigma_f_;
  double is_sigma_l_;
  Vector an_sigma_l_;
  GaussianKernel kernel_;
};

TEST_F(TestGaussianKernel, VectorConstructor) {
  const GaussianKernel kernel{Vector::Constant(3, 1.0)};
  EXPECT_EQ(kernel.sigma_f(), 1.0);
  EXPECT_EQ(kernel.sigma_l().size(), 3);
  EXPECT_THAT(kernel.sigma_l(), ::testing::Each(1.0));
  EXPECT_EQ(kernel.sigma_f(), kernel.get<double>(Parameter::SIGMA_F));
  EXPECT_EQ(kernel.sigma_l(), kernel.get<const Vector&>(Parameter::SIGMA_L));
}

TEST_F(TestGaussianKernel, DimensionConstructor) {
  const GaussianKernel kernel{4.0};
  EXPECT_EQ(kernel.sigma_f(), 1.0);
  EXPECT_EQ(kernel.sigma_l().size(), 1);
  EXPECT_THAT(kernel.sigma_l(), ::testing::Each(4.0));
  // EXPECT_EQ(kernel.sigma_f(), kernel.get<double>(Parameter::MEAN));
  EXPECT_EQ(kernel.sigma_f(), kernel.get<double>(Parameter::SIGMA_F));
  // EXPECT_EQ(kernel.sigma_l(), kernel.get<const Vector&>(Parameter::LENGTH_SCALE));
  // EXPECT_EQ(kernel.sigma_l(), kernel.get<const Vector&>(Parameter::COVARIANCE));
  EXPECT_EQ(kernel.sigma_l(), kernel.get<const Vector&>(Parameter::SIGMA_L));
}

TEST_F(TestGaussianKernel, ParametersList) {
  EXPECT_THAT(kernel_.parameters_list(),
              ::testing::UnorderedElementsAre(Parameter::SIGMA_F, Parameter::SIGMA_L, Parameter::GRADIENT_OPTIMIZABLE));
  ;
}

TEST_F(TestGaussianKernel, ParametersHas) {
  EXPECT_TRUE(kernel_.has(Parameter::SIGMA_F));
  EXPECT_TRUE(kernel_.has(Parameter::SIGMA_L));
  EXPECT_TRUE(kernel_.has(Parameter::GRADIENT_OPTIMIZABLE));
}

TEST_F(TestGaussianKernel, ParametersGet) {
  EXPECT_EQ(kernel_.get<Parameter::SIGMA_F>(), sigma_f_);
  EXPECT_EQ(kernel_.get<Parameter::SIGMA_L>(), an_sigma_l_);
  EXPECT_TRUE(kernel_.get<Parameter::GRADIENT_OPTIMIZABLE>().isApprox(an_sigma_l_.array().log().matrix()));
}

TEST_F(TestGaussianKernel, ParametersSet) {
  constexpr double new_sigma_f = 1111.0;
  const Vector new_sigma_l{Vector::Random(4)};

  EXPECT_NO_THROW(kernel_.set(Parameter::SIGMA_F, new_sigma_f));
  EXPECT_NO_THROW(kernel_.set(Parameter::SIGMA_L, new_sigma_l));

  EXPECT_EQ(kernel_.get<Parameter::SIGMA_F>(), new_sigma_f);
  EXPECT_EQ(kernel_.get<Parameter::SIGMA_L>(), new_sigma_l);

  EXPECT_NO_THROW(kernel_.set(Parameter::GRADIENT_OPTIMIZABLE, Vector{new_sigma_l * 2}));
  EXPECT_EQ(kernel_.get<Parameter::GRADIENT_OPTIMIZABLE>(), new_sigma_l * 2);
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

TEST_F(TestGaussianKernel, ApplySelfIsotropic) {
  const Vector x1{Vector::Random(4)};
  const GaussianKernel kernel{is_sigma_l_, sigma_f_};
  static_cast<void>(kernel(x1, x1));
  EXPECT_DOUBLE_EQ(kernel(x1).value(), gaussian(x1, x1, is_sigma_l_, sigma_f_).value());
}
TEST_F(TestGaussianKernel, ApplySelfAnisotropic) {
  const Vector x1{Vector::Random(4)};
  const GaussianKernel kernel{an_sigma_l_, sigma_f_};
  EXPECT_DOUBLE_EQ(kernel(x1).value(), gaussian(x1, x1, an_sigma_l_, sigma_f_).value());
}

TEST_F(TestGaussianKernel, ApplyVectorIsotropic) {
  const Vector x1{Vector::Random(4)}, x2{Vector::Random(4)};
  const GaussianKernel kernel{is_sigma_l_, sigma_f_};
  EXPECT_DOUBLE_EQ(kernel(x1, x2).value(), gaussian(x1, x2, is_sigma_l_, sigma_f_).value());
}
TEST_F(TestGaussianKernel, ApplyVectorAnisotropic) {
  const Vector x1{Vector::Random(4)}, x2{Vector::Random(4)};
  const GaussianKernel kernel{an_sigma_l_, sigma_f_};
  EXPECT_DOUBLE_EQ(kernel(x1, x2).value(), gaussian(x1, x2, an_sigma_l_, sigma_f_).value());
}

TEST_F(TestGaussianKernel, ApplyMatrixIsotropic) {
  const Matrix x1{Matrix::Random(4, 4)}, x2{Matrix::Random(3, 4)};
  const GaussianKernel kernel{is_sigma_l_, sigma_f_};
  EXPECT_TRUE(kernel(x1, x2).isApprox(gaussian(x1, x2, is_sigma_l_, sigma_f_)));
}
TEST_F(TestGaussianKernel, ApplyMatrixAnisotropic) {
  const Matrix x1{Matrix::Random(4, 4)}, x2{Matrix::Random(3, 4)};
  const GaussianKernel kernel{an_sigma_l_, sigma_f_};
  EXPECT_TRUE(kernel(x1, x2).isApprox(gaussian(x1, x2, an_sigma_l_, sigma_f_)));
}

TEST_F(TestGaussianKernel, GradientVectorIsotropic) {
  const Vector x{Vector::Random(4)};
  const GaussianKernel kernel{is_sigma_l_, sigma_f_};
  std::vector<Matrix> gradient;
  EXPECT_TRUE(kernel(x, gradient).isApprox(gaussian(x, x, is_sigma_l_, sigma_f_)));
  EXPECT_EQ(gradient.size(), 1);
  EXPECT_EQ(gradient.front().size(), 1);
  EXPECT_EQ(gradient.front().value(), 0);
}

TEST_F(TestGaussianKernel, GradientVectorAnisotropic) {
  const Vector x{Vector::Random(4)};
  const GaussianKernel kernel{an_sigma_l_, sigma_f_};
  std::vector<Matrix> gradient;
  EXPECT_TRUE(kernel(x, gradient).isApprox(gaussian(x, x, an_sigma_l_, sigma_f_)));
  EXPECT_EQ(gradient.size(), 4);
  EXPECT_EQ(gradient.front().size(), 1);
  EXPECT_EQ(gradient.front().value(), 0);
}

TEST_F(TestGaussianKernel, GradientIsotropicFixed) {
  Matrix x{4, 4};
  x << 4, 5, 6, 7, 1, 2, 3, 4, 6, 7, 8, 9, 9, 10, 11, 12;
  const GaussianKernel kernel{is_sigma_l_};
  std::vector<Matrix> gradient;
  EXPECT_TRUE(kernel(x, gradient).isApprox(gaussian(x, x, is_sigma_l_)));

  Matrix expected{4, 4};
  expected << 0, 0.274606, 0.692645, 0.0029489,  //
      0.274606, 0, 0.0029489, 9.92725e-09,       //
      0.692645, 0.0029489, 0, 0.274606,          //
      0.0029489, 9.92725e-09, 0.274606, 0;

  ASSERT_EQ(gradient.size(), 1);
  ASSERT_EQ(gradient.front().rows(), x.rows());
  ASSERT_EQ(gradient.front().cols(), x.rows());

  EXPECT_TRUE(gradient.front().isApprox(expected, 1e-6));
}

TEST_F(TestGaussianKernel, GradientAnisotropicFixed) {
  Vector sigma_l{2};
  sigma_l << 1, 2;
  Matrix x{3, 2};
  x << 4, 5, 1, 2, 6, 7;
  const GaussianKernel kernel{sigma_l};
  std::vector<Matrix> gradient;
  EXPECT_TRUE(kernel(x, gradient).isApprox(gaussian(x, x, sigma_l)));

  Matrix expected_0{3, 3}, expected_1{3, 3};
  expected_0 << 0, 0.0324591, 0.32834,  //
      0.0324591, 0, 4.09344e-06,        //
      0.32834, 4.09344e-06, 0;
  expected_1 << 0, 0.00811477, 0.082085,  //
      0.00811477, 0, 1.02336e-06,         //
      0.082085, 1.02336e-06, 0;
  ASSERT_EQ(gradient.size(), 2);
  ASSERT_EQ(gradient[0].rows(), x.rows());
  ASSERT_EQ(gradient[0].cols(), x.rows());
  ASSERT_EQ(gradient[1].rows(), x.rows());
  ASSERT_EQ(gradient[1].cols(), x.rows());

  EXPECT_TRUE(gradient[0].isApprox(expected_0, 1e-6));
  EXPECT_TRUE(gradient[1].isApprox(expected_1, 1e-6));
}
