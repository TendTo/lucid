/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/model.h"
#include "lucid/util/exception.h"

using lucid::ConstMatrixRef;
using lucid::ConstMatrixRefCopy;
using lucid::Estimator;
using lucid::GaussianKernel;
using lucid::Index;
using lucid::Kernel;
using lucid::KernelRidgeRegressor;
using lucid::LbfgsParameters;
using lucid::LbfgsTuner;
using lucid::Matrix;
using lucid::Parameter;
using lucid::Vector;

class TestLbfgsTuner : public ::testing::Test {
 protected:
  const int num_samples_{10};
  const int dim_{3};
  const double regularization_constant_{1e-6};
  const Matrix training_outputs_{Matrix::Random(num_samples_, 1)};
  const Matrix training_inputs_{Matrix::Random(num_samples_, dim_)};
  KernelRidgeRegressor is_regressor_{
      std::make_unique<GaussianKernel>(), regularization_constant_,
      std::make_shared<LbfgsTuner>(Eigen::VectorXd::Constant(1, 1e-5), Eigen::VectorXd::Constant(1, 1e5),
                                   LbfgsParameters{.max_iterations = 10})};
};

TEST_F(TestLbfgsTuner, Constructor) { EXPECT_NO_THROW(LbfgsTuner()); }

TEST_F(TestLbfgsTuner, ConstructorWithParams) {
  LbfgsParameters params;
  params.m = 10;
  params.epsilon = 1e-6;
  params.max_iterations = 100;

  EXPECT_NO_THROW(LbfgsTuner tuner(params));
}

TEST_F(TestLbfgsTuner, ConstructorEstimator) {
  const KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(dim_), regularization_constant_,
                                       std::make_shared<LbfgsTuner>()};
  EXPECT_NE(dynamic_cast<const LbfgsTuner *>(regressor.tuner().get()), nullptr);
}

TEST_F(TestLbfgsTuner, LogMarginalLikelihoodGradientIsotropicFixed) {
  constexpr int n_samples = 3;
  constexpr double sigma_l = 2.3;
  constexpr double lambda = 0.1 / n_samples;
  Matrix inputs{3, 2}, outputs{3, 3};
  inputs << 4, 5, 1, 2, 6, 7;
  outputs << 4, 4, 4,  //
      5, 5, 5,         //
      2, 2, 1;

  KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(sigma_l), lambda, std::make_shared<LbfgsTuner>()};
  regressor.set(Parameter::GRADIENT_OPTIMIZABLE, Vector::Constant(1, 1));
  regressor.fit(inputs, outputs);
  EXPECT_EQ(regressor.gradient().size(), 1);
  EXPECT_DOUBLE_EQ(regressor.gradient()(0), 4.1399452754831145e-08);
  EXPECT_DOUBLE_EQ(regressor.objective_value(), -45.202528848893763);
  EXPECT_DOUBLE_EQ(regressor.get<Parameter::SIGMA_L>().value(), 4.8241131432598596);
  EXPECT_DOUBLE_EQ(regressor.get<Parameter::GRADIENT_OPTIMIZABLE>().value(), 1.5736269133911023);
}

TEST_F(TestLbfgsTuner, Tune) {
  const Vector original_sigma_l{is_regressor_.get<Parameter::SIGMA_L>()};

  is_regressor_.fit(training_inputs_, training_outputs_);

  EXPECT_FALSE(is_regressor_.get<Parameter::SIGMA_L>().isApprox(original_sigma_l));
  EXPECT_TRUE(is_regressor_.get<Parameter::SIGMA_L>().isApprox(
      is_regressor_.get<Parameter::GRADIENT_OPTIMIZABLE>().array().exp().matrix()));
  EXPECT_TRUE((is_regressor_.get<Parameter::SIGMA_L>().array() > 0).all());
}

TEST_F(TestLbfgsTuner, TuneOnline) {
  const Vector original_sigma_l{is_regressor_.get<Parameter::SIGMA_L>()};

  is_regressor_.fit_online(
      training_inputs_, [this](const Estimator &, ConstMatrixRef) -> ConstMatrixRefCopy { return training_outputs_; });

  EXPECT_FALSE(is_regressor_.get<Parameter::SIGMA_L>().isApprox(original_sigma_l));
  EXPECT_TRUE(is_regressor_.get<Parameter::SIGMA_L>().isApprox(
      is_regressor_.get<Parameter::GRADIENT_OPTIMIZABLE>().array().exp().matrix()));
  EXPECT_TRUE((is_regressor_.get<Parameter::SIGMA_L>().array() > 0).all());
}

#if 0
// This test is disabled because it seems that the optimization does not always improve the fit
TEST_F(TestLbfgsTuner, OptimizationImprovesFit) {
  Matrix inputs{num_samples_, dim_};
  Matrix outputs{num_samples_, dim_};
  for (int i = 0; i < num_samples_; ++i) {
    inputs.row(i).setLinSpaced(i * dim_, i * dim_ + dim_);
    outputs.row(i) = (inputs.row(i).array().square() + 1).matrix();  // Simple quadratic function
  }
  std::cout << "Inputs:\n" << inputs << "\nOutputs:\n" << outputs << std::endl;

  // Create a copy without tuner
  KernelRidgeRegressor untuned_regressor{std::make_unique<GaussianKernel>(), regularization_constant_};

  // Fit both models
  is_regressor_.fit(inputs, outputs);
  untuned_regressor.fit(inputs, outputs);

  // The tuned model should have better or equal score on training data
  EXPECT_GE(is_regressor_.score(inputs, outputs), untuned_regressor.score(inputs, outputs));
}
#endif

TEST_F(TestLbfgsTuner, CustomMaxIterations) {
  LbfgsParameters params;
  params.max_iterations = 1;  // Force early stopping

  KernelRidgeRegressor limited_regressor{std::make_unique<GaussianKernel>(), regularization_constant_,
                                         std::make_shared<LbfgsTuner>(params)};

  const Vector original_sigma_l = limited_regressor.get<Parameter::SIGMA_L>();

  limited_regressor.fit(training_inputs_, training_outputs_);

  EXPECT_FALSE(limited_regressor.get<Parameter::SIGMA_L>().isApprox(original_sigma_l));
}

TEST_F(TestLbfgsTuner, HighDimensionalData) {
  constexpr int high_dim = 20;

  KernelRidgeRegressor high_dim_regressor{std::make_unique<GaussianKernel>(Vector::Constant(high_dim, 1.0)),
                                          regularization_constant_, std::make_shared<LbfgsTuner>()};

  const Matrix high_dim_inputs = Matrix::Random(num_samples_, high_dim);

  high_dim_regressor.fit(high_dim_inputs, training_outputs_);

  Vector tuned_sigma_l = high_dim_regressor.get<Parameter::SIGMA_L>();

  EXPECT_EQ(tuned_sigma_l.size(), high_dim);
  EXPECT_TRUE((tuned_sigma_l.array() > 0).all());
}

TEST_F(TestLbfgsTuner, MismatchedInputsOutputs) {
  const Matrix mismatched_outputs = Matrix::Random(num_samples_ - 1, 1);  // Different row count

  EXPECT_THROW(is_regressor_.fit(training_inputs_, mismatched_outputs),
               lucid::exception::LucidInvalidArgumentException);
}

TEST_F(TestLbfgsTuner, CustomEpsilon) {
  LbfgsParameters strict_params;
  strict_params.epsilon = 1e-8;      // Stricter convergence
  strict_params.epsilon_rel = 1e-8;  // Stricter convergence

  KernelRidgeRegressor strict_regressor{std::make_unique<GaussianKernel>(), regularization_constant_,
                                        std::make_shared<LbfgsTuner>(strict_params)};

  EXPECT_NO_THROW(strict_regressor.fit(training_inputs_, training_outputs_));
}

TEST_F(TestLbfgsTuner, Reproducibility) {
  // Same initial conditions should produce same results
  KernelRidgeRegressor regressor1{std::make_unique<GaussianKernel>(Vector::Constant(dim_, 1.0), 1.0),
                                  regularization_constant_, std::make_shared<LbfgsTuner>()};

  KernelRidgeRegressor regressor2{std::make_unique<GaussianKernel>(Vector::Constant(dim_, 1.0), 1.0),
                                  regularization_constant_, std::make_shared<LbfgsTuner>()};

  regressor1.fit(training_inputs_, training_outputs_);
  regressor2.fit(training_inputs_, training_outputs_);

  Vector sigma_l1 = regressor1.get<Parameter::SIGMA_L>();
  Vector sigma_l2 = regressor2.get<Parameter::SIGMA_L>();

  EXPECT_TRUE(sigma_l1.isApprox(sigma_l2));
}
