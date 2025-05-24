/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/MedianHeuristicTuner.h"
#include "lucid/model/model.h"
#include "lucid/util/exception.h"

using lucid::GaussianKernel;
using lucid::GridSearchTuner;
using lucid::Index;
using lucid::Kernel;
using lucid::KernelRidgeRegressor;
using lucid::Matrix;
using lucid::Parameter;
using lucid::ParameterValues;
using lucid::Vector;

class TestGridSearchTuner : public ::testing::Test {
 protected:
  const int num_samples_{10};
  const int dim_{3};
  const double sigma_f_{0.0};
  const double sigma_l_{0.0};
  const double regularization_constant_{1e-6};
  const Matrix training_outputs_{Matrix::Random(num_samples_, 1)};
  const std::vector<ParameterValues> parameters_{
      ParameterValues{Parameter::SIGMA_L, std::vector<Vector>{Vector::Constant(dim_, 0.1)}},
      ParameterValues{Parameter::SIGMA_F, 0.1, 1.0, 10.1},
      ParameterValues{Parameter::REGULARIZATION_CONSTANT, 1e-6, 1e-2, 10}};
  KernelRidgeRegressor regressor_{std::make_unique<GaussianKernel>(dim_, sigma_l_, sigma_f_), regularization_constant_,
                                  std::make_shared<GridSearchTuner>(parameters_)};
};

TEST_F(TestGridSearchTuner, Constructor) { EXPECT_NO_THROW(GridSearchTuner{parameters_}); }

TEST_F(TestGridSearchTuner, ConstructorEstimator) {
  const KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(3), 0,
                                       std::make_shared<GridSearchTuner>(parameters_)};
  EXPECT_NE(dynamic_cast<GridSearchTuner *>(regressor.tuner().get()), nullptr);
}

TEST_F(TestGridSearchTuner, Tune) {
  const Matrix training_inputs{Matrix::Random(num_samples_, dim_)};

  regressor_.fit(training_inputs, training_outputs_);

  EXPECT_EQ(regressor_.get<double>(Parameter::REGULARIZATION_CONSTANT), regularization_constant_);
  EXPECT_EQ(regressor_.get<double>(Parameter::SIGMA_F), sigma_f_);
  EXPECT_NE(regressor_.get<const Vector &>(Parameter::SIGMA_L), Vector::Constant(dim_, sigma_l_));
  EXPECT_TRUE((regressor_.get<const Vector &>(Parameter::SIGMA_L).array() > 0).all());
}
