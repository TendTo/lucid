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
using lucid::Estimator;
using lucid::GaussianKernel;
using lucid::Index;
using lucid::Kernel;
using lucid::KernelRidgeRegressor;
using lucid::Matrix;
using lucid::Parameter;
using lucid::RectSet;
using lucid::Scalar;
using lucid::Vector;
using lucid::Vector2;

class TestKernelRidgeRegressor : public ::testing::Test {
 protected:
  const Dimension n_samples_{50};               //< Number of samples to collect
  const Dimension dim_{1};                      //< State space dimension
  const double sigma_f_{1.0};                   //< Kernel amplitude
  const double sigma_l_{2.0};                   //< Kernel length scale
  const double regularization_constant_{1e-6};  //< Regularization constant for the kernel ridge regressor
  const RectSet x_limits_{std::vector<std::pair<double, double>>(dim_, {-1.0, 1.0})};  //< Limits of the input space
  KernelRidgeRegressor regressor_{std::make_unique<GaussianKernel>(dim_, sigma_l_, sigma_f_), regularization_constant_};

  [[nodiscard]] std::pair<KernelRidgeRegressor, ConstantTruncatedFourierFeatureMap> get_regression_and_feature_map(
      const double sigma_l, const int num_frequencies) const {
    const Matrix training_inputs{Matrix::Random(n_samples_, dim_)};
    const Matrix training_outputs{Matrix::Random(n_samples_, dim_)};
    KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(dim_, sigma_l, sigma_f_)};
    regressor.fit(training_inputs, training_outputs);
    return {std::move(regressor), ConstantTruncatedFourierFeatureMap(num_frequencies, sigma_l, sigma_f_, x_limits_)};
  }

  [[nodiscard]] std::pair<Matrix, Matrix> getTrainingData() const {
    return {Matrix::Random(n_samples_, dim_), Matrix::Random(n_samples_, dim_)};
  }
};

TEST_F(TestKernelRidgeRegressor, FitAndPredict) {
  auto [inputs, outputs] = getTrainingData();
  EXPECT_NO_THROW(regressor_.fit(inputs, outputs));

  Matrix test_inputs{Matrix::Random(10, dim_)};
  Matrix predictions;
  EXPECT_NO_THROW(predictions = regressor_.predict(test_inputs));

  EXPECT_EQ(predictions.rows(), test_inputs.rows());
  EXPECT_EQ(predictions.cols(), outputs.cols());
}

TEST_F(TestKernelRidgeRegressor, ParameterHas) {
  ASSERT_TRUE(regressor_.has(Parameter::REGULARIZATION_CONSTANT));
  ASSERT_TRUE(regressor_.has(Parameter::SIGMA_F));
  ASSERT_TRUE(regressor_.has(Parameter::SIGMA_L));
}

TEST_F(TestKernelRidgeRegressor, ParameterGet) {
  EXPECT_EQ(regressor_.get<double>(Parameter::REGULARIZATION_CONSTANT), regularization_constant_);
  EXPECT_EQ(regressor_.get<double>(Parameter::SIGMA_F), sigma_f_);
  EXPECT_EQ(regressor_.get<const Vector&>(Parameter::SIGMA_L), Vector::Constant(dim_, sigma_l_));
}

TEST_F(TestKernelRidgeRegressor, ParameterSet) {
  constexpr double new_reg = 100;
  constexpr double new_sigma_f = 1111.0;
  const Vector new_sigma_l{Vector::Random(dim_)};

  regressor_.set(Parameter::REGULARIZATION_CONSTANT, new_reg);
  regressor_.set(Parameter::SIGMA_F, new_sigma_f);
  regressor_.set(Parameter::SIGMA_L, new_sigma_l);

  EXPECT_EQ(regressor_.get<double>(Parameter::REGULARIZATION_CONSTANT), new_reg);
  EXPECT_EQ(regressor_.get<double>(Parameter::SIGMA_F), new_sigma_f);
  EXPECT_EQ(regressor_.get<const Vector&>(Parameter::SIGMA_L), new_sigma_l);
}

TEST_F(TestKernelRidgeRegressor, HiddenKernelParameter) {
  class HiddenKernel final : public Kernel {
   public:
    [[nodiscard]] bool is_stationary() const override { return true; }
    Scalar operator()(const Vector&, const Vector&) const override { return 0.0; }
    [[nodiscard]] std::unique_ptr<Kernel> clone() const override { return std::make_unique<HiddenKernel>(); }
    [[nodiscard]] bool has(Parameter parameter) const override {
      return parameter == Parameter::REGULARIZATION_CONSTANT;
    }
  };

  EXPECT_THROW(KernelRidgeRegressor(std::make_unique<HiddenKernel>(), 1.0),
               lucid::exception::LucidInvalidArgumentException);
}

TEST_F(TestKernelRidgeRegressor, Clone) {
  auto [inputs, outputs] = getTrainingData();
  regressor_.fit(inputs, outputs);

  std::unique_ptr<Estimator> clone_ptr;
  EXPECT_NO_THROW(clone_ptr = regressor_.clone());
  EXPECT_NE(clone_ptr.get(), nullptr);
  EXPECT_NE(clone_ptr.get(), &regressor_);

  const auto& clone = static_cast<const KernelRidgeRegressor&>(*clone_ptr.get());

  EXPECT_EQ(clone.get<double>(Parameter::REGULARIZATION_CONSTANT), regressor_.regularization_constant());
  EXPECT_EQ(clone.get<double>(Parameter::SIGMA_F), regressor_.get<double>(Parameter::SIGMA_F));
  EXPECT_EQ(clone.get<const Vector&>(Parameter::SIGMA_L), regressor_.get<const Vector&>(Parameter::SIGMA_L));
  EXPECT_NE(clone.kernel().get(), regressor_.kernel().get());

  Matrix test_inputs{Matrix::Random(10, dim_)};
  Matrix original_predictions = regressor_.predict(test_inputs);
  Matrix cloned_predictions = clone_ptr->predict(test_inputs);

  EXPECT_EQ(original_predictions, cloned_predictions);
}

TEST_F(TestKernelRidgeRegressor, ErrorHandling) {
  auto [inputs, outputs] = getTrainingData();

  // Predict before fitting
  const Matrix test_inputs{Matrix::Random(10, dim_)};
  EXPECT_THROW(static_cast<void>(regressor_.predict(test_inputs)), lucid::exception::LucidInvalidArgumentException);

  // Fit the model
  regressor_.fit(inputs, outputs);

  // Predict with incorrect dimensions
  const Matrix wrong_dim_inputs{Matrix::Random(10, dim_ + 1)};
  EXPECT_THROW(static_cast<void>(regressor_.predict(wrong_dim_inputs)),
               lucid::exception::LucidInvalidArgumentException);

  // Fit with mismatched inputs/outputs
  const Matrix mismatched_outputs{Matrix::Random(n_samples_ + 1, dim_)};
  EXPECT_THROW(regressor_.fit(inputs, mismatched_outputs), lucid::exception::LucidInvalidArgumentException);
}

TEST_F(TestKernelRidgeRegressor, FeatureMapPrediction) {
  const auto [inputs, outputs] = getTrainingData();
  regressor_.fit(inputs, outputs);

  ConstantTruncatedFourierFeatureMap feature_map(10, sigma_l_, sigma_f_, x_limits_);
  const Matrix predictions = regressor_.predict(inputs);
  const Matrix fm_predictions = regressor_.predict(inputs, feature_map);

  EXPECT_EQ(fm_predictions.rows(), predictions.rows());
  EXPECT_EQ(fm_predictions.cols(), predictions.cols());
}

TEST_F(TestKernelRidgeRegressor, MultipleFeatures) {
  constexpr int input_dim = 3;
  constexpr int output_dim = 2;
  const Matrix inputs{Matrix::Random(n_samples_, input_dim)};
  const Matrix outputs{Matrix::Random(n_samples_, output_dim)};

  KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(input_dim, sigma_l_, sigma_f_),
                                 regularization_constant_};

  const Matrix test_inputs{Matrix::Random(n_samples_ + 1, input_dim)};
  regressor.fit(inputs, outputs);
  const Matrix predictions = regressor.predict(test_inputs);

  EXPECT_EQ(predictions.rows(), test_inputs.rows());
  EXPECT_EQ(predictions.cols(), output_dim);
}

TEST_F(TestKernelRidgeRegressor, SimpleInterpolation) {
  // Create linear function y = 2x + 1
  Matrix inputs = Matrix::Zero(5, 1);
  Matrix outputs = Matrix::Zero(5, 1);

  inputs << -1, -0.5, 0, 0.5, 1;
  outputs << -1, 0, 1, 2, 3;

  KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(1, 0.3, 1.0), 1e-10};
  regressor.fit(inputs, outputs);

  // Test interpolation at 0.25
  Matrix test_point(1, 1);
  test_point << 0.25;

  // Expected value is 2*0.25 + 1 = 1.5
  EXPECT_NEAR(regressor.predict(test_point).value(), 1.5, 0.1);
}