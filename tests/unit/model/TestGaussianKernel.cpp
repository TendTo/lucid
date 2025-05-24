/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "lucid/model/GaussianKernel.h"

using lucid::GaussianKernel;
using lucid::Matrix;
using lucid::Parameter;
using lucid::Vector;

class TestGaussianKernel : public ::testing::Test {
 protected:
  static Vector get_sigma_l() {
    Vector sigma_l{4};
    sigma_l << 3.2, 5.1, 3.4, 1.24;
    return sigma_l;
  }

  TestGaussianKernel() : sigma_f_{2.01}, sigma_l_{get_sigma_l()}, kernel_{get_sigma_l(), sigma_f_} {}

  double sigma_f_;
  Vector sigma_l_{};
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

TEST_F(TestGaussianKernel, ApplyVector) {
  Vector x1{4}, x2{4};
  x1 << 4, 5, 6, 7;
  x2 << 1, 2, 3, 4;
  // (x1 / sigma_l) - (x2 / sigma_l)
  const auto diff{x1.cwiseProduct(sigma_l_.cwiseInverse()) - x2.cwiseProduct(sigma_l_.cwiseInverse())};
  // ||(x1 / sigma_l) - (x2 / sigma_l)||^2
  const double distance = diff.squaredNorm();
  // sigma_f^2 * exp(-0.5 * ||(x1 / sigma_l) - (x2 / sigma_l)||^2)
  const double expected = sigma_f_ * sigma_f_ * std::exp(-0.5 * distance);
  EXPECT_DOUBLE_EQ(kernel_(x1, x2), expected);
}

// TEST_F(TestGaussianKernel, ApplyMatrix) {
//   Matrix x1{4, 4}, x2{4, 3};
//   x1 << 4, 5, 6, 7,  //
//       1, 2, 3, 4,    //
//       6, 7, 8, 9,    //
//       9, 10, 11, 12;
//   x2 << 1, 2, 3,  //
//       4, 5, 6,    //
//       4, 4, 2,    //
//       1, 1, 1;
//   // ||(x1 / sigma_l) - (x2 / sigma_l)||^2
//   for (lucid::Index i = 0; i < x1.rows(); i++) {
//     x1.row(i) = x1.row(i).cwiseProduct(sigma_l_.transpose().cwiseInverse()).eval();
//   }
//   for (lucid::Index i = 0; i < x2.rows(); i++) {
//     x2.row(i) = x2.row(i).cwiseProduct(sigma_l_.transpose().cwiseInverse()).eval();
//   }
//   const Matrix distance = lucid::pdist<2, true>(x1, x2);
//   // sigma_f^2 * exp(-0.5 * ||(x1 / sigma_l) - (x2 / sigma_l)||^2)
//   const Matrix expected = sigma_f_ * sigma_f_ * (-0.5 * distance.array()).exp();
//   std::cout << "Expected: \n" << expected << std::endl;
//   EXPECT_EQ(kernel_.apply(x1, x2), expected);
// }
