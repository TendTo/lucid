/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/ConstantTruncatedFourierFeatureMap.h"
#include "lucid/model/GaussianKernel.h"
#include "lucid/model/KernelRidgeRegressor.h"
#include "lucid/model/RectSet.h"

using lucid::ConstantTruncatedFourierFeatureMap;
using lucid::Dimension;
using lucid::GaussianKernel;
using lucid::Index;
using lucid::Kernel;
using lucid::KernelRidgeRegressor;
using lucid::Matrix;
using lucid::RectSet;
using lucid::Scalar;
using lucid::Vector;
using lucid::Vector2;

class TestKernelRidgeRegression : public ::testing::Test {
 protected:
  const Dimension n_samples_{50};               //< Number of samples to collect
  const Dimension dim_{1};                      //< State space dimension
  const double sigma_f_{1.0};                   //< Kernel amplitude
  const double regularization_constant_{1e-6};  //< Regularization constant for the kernel ridge regressor
  const RectSet x_limits_{std::vector<std::pair<double, double>>(dim_, {-1.0, 1.0})};  //< Limits of the input space

  [[nodiscard]] std::pair<KernelRidgeRegressor, ConstantTruncatedFourierFeatureMap> get_regression_and_feature_map(
      const double sigma_l, const int num_frequencies) const {
    const Matrix training_inputs{Matrix::Random(n_samples_, dim_)};
    const Matrix training_outputs{Matrix::Random(n_samples_, dim_)};
    KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(dim_, sigma_l, sigma_f_)};
    regressor.fit(training_inputs, training_outputs);
    return {std::move(regressor), ConstantTruncatedFourierFeatureMap(num_frequencies, sigma_l, sigma_f_, x_limits_)};
  }
};

TEST_F(TestKernelRidgeRegression, Contains) {
  constexpr std::array<double, 4> sigma_ls{0.1, 0.5, 1.0, 2.0};
  constexpr std::array<int, 4> nums_frequencies{1, 2, 3, 4};

  for (const double sigma_l : sigma_ls) {
    for (const int num_frequencies : nums_frequencies) {
      const auto [regressor, feature_map] = get_regression_and_feature_map(sigma_l, num_frequencies);
      const Matrix interpolation = regressor(Matrix::Random(n_samples_, dim_));
      const Matrix truncated_interpolation = regressor(Matrix::Random(n_samples_, dim_), feature_map);

      EXPECT_EQ(interpolation.rows(), n_samples_);
      EXPECT_EQ(interpolation.cols(), dim_);
    }
  }
}
