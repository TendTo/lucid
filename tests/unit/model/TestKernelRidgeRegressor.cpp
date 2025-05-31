/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numbers>

#include "lucid/model/ConstantTruncatedFourierFeatureMap.h"
#include "lucid/model/GaussianKernel.h"
#include "lucid/model/GramMatrix.h"
#include "lucid/model/KernelRidgeRegressor.h"
#include "lucid/model/RectSet.h"

using lucid::ConstantTruncatedFourierFeatureMap;
using lucid::ConstMatrixRef;
using lucid::Dimension;
using lucid::Estimator;
using lucid::GaussianKernel;
using lucid::Index;
using lucid::Kernel;
using lucid::KernelRidgeRegressor;
using lucid::Matrix;
using lucid::Parameter;
using lucid::Parameters;
using lucid::RectSet;
using lucid::Request;
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
  KernelRidgeRegressor regressor_{std::make_unique<GaussianKernel>(sigma_l_, sigma_f_), regularization_constant_};

  [[nodiscard]] std::pair<KernelRidgeRegressor, ConstantTruncatedFourierFeatureMap> get_regression_and_feature_map(
      const double sigma_l, const int num_frequencies) const {
    const Matrix training_inputs{Matrix::Random(n_samples_, dim_)};
    const Matrix training_outputs{Matrix::Random(n_samples_, dim_)};
    KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(sigma_l, sigma_f_)};
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

  const Matrix test_inputs{Matrix::Random(10, dim_)};
  Matrix predictions;
  EXPECT_NO_THROW(predictions = regressor_.predict(test_inputs));

  EXPECT_EQ(predictions.rows(), test_inputs.rows());
  EXPECT_EQ(predictions.cols(), outputs.cols());
}

TEST_F(TestKernelRidgeRegressor, ParametersList) {
  EXPECT_THAT(regressor_.parameters_list(),
              ::testing::UnorderedElementsAre(Parameter::SIGMA_F, Parameter::SIGMA_L,
                                              Parameter::REGULARIZATION_CONSTANT, Parameter::GRADIENT_OPTIMIZABLE));
}

TEST_F(TestKernelRidgeRegressor, ParameterHas) {
  EXPECT_TRUE(regressor_.has(Parameter::REGULARIZATION_CONSTANT));
  EXPECT_TRUE(regressor_.has(Parameter::SIGMA_F));
  EXPECT_TRUE(regressor_.has(Parameter::SIGMA_L));
}

TEST_F(TestKernelRidgeRegressor, ParameterGet) {
  EXPECT_EQ(regressor_.get<Parameter::REGULARIZATION_CONSTANT>(), regularization_constant_);
  EXPECT_EQ(regressor_.get<Parameter::SIGMA_F>(), sigma_f_);
  EXPECT_EQ(regressor_.get<Parameter::SIGMA_L>(), Vector::Constant(dim_, sigma_l_));
}

TEST_F(TestKernelRidgeRegressor, ParameterSet) {
  constexpr double new_reg = 100;
  constexpr double new_sigma_f = 1111.0;
  const Vector new_sigma_l{Vector::Random(dim_)};

  regressor_.set(Parameter::REGULARIZATION_CONSTANT, new_reg);
  regressor_.set(Parameter::SIGMA_F, new_sigma_f);
  regressor_.set(Parameter::SIGMA_L, new_sigma_l);

  EXPECT_EQ(regressor_.get<Parameter::REGULARIZATION_CONSTANT>(), new_reg);
  EXPECT_EQ(regressor_.get<Parameter::SIGMA_F>(), new_sigma_f);
  EXPECT_EQ(regressor_.get<Parameter::SIGMA_L>(), new_sigma_l);
}

TEST_F(TestKernelRidgeRegressor, HiddenKernelParameter) {
  class HiddenKernel final : public Kernel {
   public:
    HiddenKernel() : Kernel{static_cast<Parameters>(Parameter::REGULARIZATION_CONSTANT)} {}
    [[nodiscard]] bool is_stationary() const override { return true; }
    Matrix operator()(ConstMatrixRef, const ConstMatrixRef&, std::vector<Matrix>*) const override { return {}; }
    [[nodiscard]] std::unique_ptr<Kernel> clone() const override { return std::make_unique<HiddenKernel>(); }
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

  KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(sigma_l_, sigma_f_), regularization_constant_};

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

  KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(0.3, 1.0), 1e-10};
  regressor.fit(inputs, outputs);

  // Test interpolation at 0.25
  Matrix test_point(1, 1);
  test_point << 0.25;

  // Expected value is 2*0.25 + 1 = 1.5
  EXPECT_NEAR(regressor.predict(test_point).value(), 1.5, 0.1);
}

TEST_F(TestKernelRidgeRegressor, LogMarginalLikelihood) {
  constexpr int n_samples = 15;
  constexpr double lambda = 0.2 / n_samples;
  const Matrix inputs{Matrix::Random(n_samples, 2)};
  const Matrix outputs{Matrix::Random(n_samples, 3)};

  const GaussianKernel kernel{1, 1};
  lucid::GramMatrix K{kernel, inputs};
  K.add_diagonal_term(lambda * static_cast<double>(n_samples));  // Add regularisation term to the diagonal
  const Matrix w{K.inverse() * outputs};
  double expected = (-0.5 * (outputs.array() * w.array()).matrix().colwise().sum().array()  //  -0.5 . (y^T * w)
                     - K.L().diagonal().array().log().sum()                                 //  -sum(log(diag(L)))
                     - std::log(2 * std::numbers::pi) * outputs.rows() / 2)                 // -log(2*pi) * n / 2
                        .sum();

  KernelRidgeRegressor regressor{kernel, lambda};
  regressor.consolidate(inputs, outputs, Request::OBJECTIVE_VALUE | Request::_);
  const double log_likelihood = regressor.log_marginal_likelihood();

  EXPECT_DOUBLE_EQ(log_likelihood, expected);
}

TEST_F(TestKernelRidgeRegressor, LogMarginalLikelihoodFixed) {
  constexpr int n_samples = 10;
  constexpr double lambda = 0.1 / n_samples;
  Matrix inputs{n_samples, 2};
  Matrix outputs{n_samples, 3};
  inputs << 1.87270059, 4.75357153, 3.65996971, 2.99329242, 0.7800932, 0.7799726, 0.29041806, 4.33088073, 3.00557506,
      3.54036289, 0.10292247, 4.84954926, 4.1622132, 1.06169555, 0.90912484, 0.91702255, 1.52121121, 2.62378216,
      2.15972509, 1.4561457;
  outputs << 0.26477707, -0.05268811, 0.16274203, -0.49547106, 0.14775724, -0.86862445, 0.70334567, 0.70325994,
      0.71084799, 0.2863528, -0.92810414, 0.95812425, 0.13559858, -0.38828536, -0.99076386, -0.38710105, -0.9368763,
      0.27899504, -0.85243263, 0.87318314, -0.52283707, 0.78896631, 0.7937943, 0.61443646, 0.99877091, 0.49497885,
      0.0495648, 0.83153619, 0.99343481, -0.55547057;

  KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(1, 1), lambda};
  regressor.fit(inputs, outputs);
  const double log_likelihood = regressor.log_marginal_likelihood();

  EXPECT_DOUBLE_EQ(log_likelihood, -28.5091519732376);
}

TEST_F(TestKernelRidgeRegressor, LogMarginalLikelihoodGradientIsotropicFixed) {
  LUCID_LOG_INIT_VERBOSITY(1);
  constexpr int n_samples = 3;
  constexpr double lambda = 0.1 / n_samples;
  Matrix inputs{3, 2}, outputs{3, 3};
  inputs << 4, 5, 1, 2, 6, 7;
  outputs << 4, 4, 4,  //
      5, 5, 5,         //
      2, 2, 1;

  const GaussianKernel kernel{2, 1};
  lucid::GramMatrix K{kernel, inputs};

  KernelRidgeRegressor regressor{kernel, lambda};
  regressor.fit(inputs, outputs);
  FAIL();
}
TEST_F(TestKernelRidgeRegressor, LogMarginalLikelihoodGradientAnisotropicFixed) {
  LUCID_LOG_INIT_VERBOSITY(1);
  constexpr int n_samples = 3;
  constexpr double lambda = 0.1 / n_samples;
  Vector sigma_l{2};
  Matrix inputs{3, 2}, outputs{3, 3};
  sigma_l << 1, 2;
  inputs << 4, 5, 1, 2, 6, 7;
  outputs << 4, 4, 4,  //
      5, 5, 5,         //
      2, 2, 1;

  const GaussianKernel kernel{sigma_l};
  lucid::GramMatrix K{kernel, inputs};

  KernelRidgeRegressor regressor{kernel, lambda};
  regressor.fit(inputs, outputs);
  FAIL();
}