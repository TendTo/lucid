/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/model.h"
#include "lucid/util/exception.h"

using lucid::GaussianKernel;
using lucid::Index;
using lucid::Kernel;
using lucid::KernelRidgeRegressor;
using lucid::Matrix;
using lucid::MedianHeuristicTuner;
using lucid::Parameter;
using lucid::Vector;

class TestMedianHeuristicTuner : public ::testing::Test {
 protected:
  const int num_samples_{10};
  const int dim_{3};
  const double sigma_f_{0.0};
  const double sigma_l_{0.0};
  const double regularization_constant_{1e-6};
  const Matrix training_outputs_{Matrix::Random(num_samples_, 1)};
  KernelRidgeRegressor regressor_{std::make_unique<GaussianKernel>(dim_, sigma_l_, sigma_f_), regularization_constant_,
                                  std::make_shared<MedianHeuristicTuner>()};
};

TEST_F(TestMedianHeuristicTuner, Constructor) { EXPECT_NO_THROW(MedianHeuristicTuner()); }

TEST_F(TestMedianHeuristicTuner, ConstructorEstimator) {
  KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(3), 0, std::make_shared<MedianHeuristicTuner>()};
  EXPECT_NE(dynamic_cast<MedianHeuristicTuner *>(regressor.tuner().get()), nullptr);
}

TEST_F(TestMedianHeuristicTuner, Tune) {
  const Matrix training_inputs{Matrix::Random(num_samples_, dim_)};

  regressor_.fit(training_inputs, training_outputs_);

  EXPECT_EQ(regressor_.get<double>(Parameter::REGULARIZATION_CONSTANT), regularization_constant_);
  EXPECT_EQ(regressor_.get<double>(Parameter::SIGMA_F), sigma_f_);
  EXPECT_NE(regressor_.get<const Vector &>(Parameter::SIGMA_L), Vector::Constant(dim_, sigma_l_));
  EXPECT_TRUE((regressor_.get<const Vector &>(Parameter::SIGMA_L).array() > 0).all());
}

TEST_F(TestMedianHeuristicTuner, TuneUniformData) {
  Matrix training_inputs(num_samples_, dim_);
  for (int i = 0; i < num_samples_; ++i) {
    training_inputs(i, 0) = i / static_cast<double>(num_samples_ - 1);
    training_inputs(i, 1) = i / static_cast<double>(num_samples_ - 1);
    training_inputs(i, 2) = i / static_cast<double>(num_samples_ - 1);
  }

  regressor_.fit(training_inputs, training_outputs_);

  // For uniform grid, the median distances should be consistent
  Vector tuned_sigma_l = regressor_.get<const Vector &>(Parameter::SIGMA_L);
  EXPECT_DOUBLE_EQ(tuned_sigma_l(0), tuned_sigma_l(1));
  EXPECT_DOUBLE_EQ(tuned_sigma_l(1), tuned_sigma_l(2));
}

TEST_F(TestMedianHeuristicTuner, DifferentScales) {
  constexpr int dim = 2;

  KernelRidgeRegressor regressor(std::make_unique<GaussianKernel>(dim), 1e-6, std::make_shared<MedianHeuristicTuner>());

  Matrix training_inputs(num_samples_, dim);
  for (int i = 0; i < num_samples_; ++i) {
    training_inputs(i, 0) = i / static_cast<double>(num_samples_ - 1);         // Range [0, 1]
    training_inputs(i, 1) = 10 * (i / static_cast<double>(num_samples_ - 1));  // Range [0, 10]
  }

  regressor.fit(training_inputs, training_outputs_);

  // The length scales should reflect the different scales of the input dimensions
  Vector tuned_sigma_l = regressor.get<const Vector &>(Parameter::SIGMA_L);
  EXPECT_GT(tuned_sigma_l(1), tuned_sigma_l(0));
  EXPECT_NEAR(tuned_sigma_l(1) / tuned_sigma_l(0), 10.0, 5.0);  // Approximate ratio of scales
}

TEST_F(TestMedianHeuristicTuner, MismatchedInputsOutputs) {
  const Matrix training_inputs = Matrix::Random(10, dim_);
  const Matrix training_outputs = Matrix::Random(5, dim_);  // Different number of rows

  EXPECT_THROW(regressor_.fit(training_inputs, training_outputs), lucid::exception::LucidInvalidArgumentException);
}

TEST_F(TestMedianHeuristicTuner, HighDimensionalData) {
  constexpr int num_samples = 50;
  constexpr int dim = 10;

  const auto custom_regressor = std::make_unique<KernelRidgeRegressor>(std::make_unique<GaussianKernel>(dim), 1e-6,
                                                                       std::make_shared<MedianHeuristicTuner>());

  const Matrix training_inputs = Matrix::Random(num_samples, dim);
  const Matrix training_outputs = Matrix::Random(num_samples, 1);

  custom_regressor->fit(training_inputs, training_outputs);

  Vector tuned_sigma_l = custom_regressor->get<const Vector &>(Parameter::SIGMA_L);
  EXPECT_EQ(tuned_sigma_l.size(), dim);
  for (int i = 0; i < dim; ++i) {
    EXPECT_GT(tuned_sigma_l(i), 0.0);
  }
}

TEST_F(TestMedianHeuristicTuner, SingleSample) {
  const Matrix training_inputs = Matrix::Random(1, dim_);
  const Matrix training_outputs = Matrix::Random(1, 1);

  EXPECT_THROW(regressor_.fit(training_inputs, training_outputs), lucid::exception::LucidInvalidArgumentException);
}

TEST_F(TestMedianHeuristicTuner, IdenticalSamples) {
  constexpr int num_samples = 10;
  constexpr int dim = 3;

  Matrix training_inputs(num_samples, dim);
  for (int i = 0; i < num_samples; ++i) {
    training_inputs.row(i) << 1.0, 2.0, 3.0;  // All rows are identical
  }

  regressor_.fit(training_inputs, training_outputs_);

  Vector tuned_sigma_l = regressor_.get<Parameter::SIGMA_L>();
  for (int i = 0; i < dim; ++i) {
    EXPECT_DOUBLE_EQ(tuned_sigma_l(i), 0.0);
  }
}