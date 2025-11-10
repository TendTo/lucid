/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>

#include "lucid/model/GridSearchTuner.h"
#include "lucid/model/Scorer.h"
#include "lucid/model/model.h"
#include "lucid/util/IndexIterator.h"
#include "lucid/util/error.h"
#include "lucid/util/exception.h"

using lucid::ConstantTruncatedFourierFeatureMap;
using lucid::ConstMatrixRef;
using lucid::ConstMatrixRefCopy;
using lucid::Estimator;
using lucid::GaussianKernel;
using lucid::GridSearchTuner;
using lucid::Index;
using lucid::Kernel;
using lucid::KernelRidgeRegressor;
using lucid::LinearTruncatedFourierFeatureMap;
using lucid::LogTruncatedFourierFeatureMap;
using lucid::Matrix;
using lucid::Parameter;
using lucid::ParameterValues;
using lucid::RectSet;
using lucid::Requests;
using lucid::Vector;
using lucid::scorer::r2_score;

class MockEstimator_ : public Estimator {
 public:
  explicit MockEstimator_(Matrix predictions, Vector sigma_l, const double sigma_f,
                          const double regularization_constant, const int degree)
      : Estimator{Parameter::DEGREE | Parameter::REGULARIZATION_CONSTANT | Parameter::SIGMA_F | Parameter::SIGMA_L},
        predictions_{std::move(predictions)},
        expected_sigma_l_(std::move(sigma_l)),
        expected_sigma_f_{sigma_f},
        expected_regularization_constant_{regularization_constant},
        expected_degree_{degree} {
    ON_CALL(*this, predict).WillByDefault(testing::Invoke([this](ConstMatrixRef) -> Matrix {
      if (sigma_f_ == expected_sigma_f_ && sigma_l_ == expected_sigma_l_ &&
          regularization_constant_ == expected_regularization_constant_ && degree_ == expected_degree_)
        return predictions_;
      return predictions_.array() + 1;  // Alter predictions if parameters are not as expected
    }));
    ON_CALL(*this, consolidate_impl).WillByDefault(testing::ReturnRef(*this));
    ON_CALL(*this, score).WillByDefault(testing::Invoke([this](ConstMatrixRef inputs, ConstMatrixRef outputs) {
      return r2_score(*this, inputs, outputs);
    }));
    ON_CALL(*this, set(testing::An<Parameter>(), testing::An<int>()))
        .WillByDefault(testing::Invoke([this](const Parameter parameter, const int value) {
          ASSERT_EQ(parameter, Parameter::DEGREE);
          degree_ = value;
        }));
    ON_CALL(*this, set(testing::An<Parameter>(), testing::An<double>()))
        .WillByDefault(testing::Invoke([this](const Parameter parameter, const double value) {
          ASSERT_TRUE(parameter == Parameter::SIGMA_F || parameter == Parameter::REGULARIZATION_CONSTANT);
          if (parameter == Parameter::REGULARIZATION_CONSTANT)
            regularization_constant_ = value;
          else
            sigma_f_ = value;
        }));
    ON_CALL(*this, set(testing::An<Parameter>(), testing::An<const Vector &>()))
        .WillByDefault(testing::Invoke([this](const Parameter parameter, const Vector &value) {
          ASSERT_EQ(parameter, Parameter::SIGMA_L);
          sigma_l_ = value;
        }));
    ON_CALL(*this, get_d).WillByDefault(testing::Invoke([this](const Parameter parameter) {
      LUCID_ASSERT(parameter == Parameter::SIGMA_F || parameter == Parameter::REGULARIZATION_CONSTANT,
                   "Parameter must be either SIGMA_F or REGULARIZATION_CONSTANT");
      return parameter == Parameter::SIGMA_F ? sigma_f_ : regularization_constant_;
    }));
    ON_CALL(*this, get_i).WillByDefault(testing::Return(degree));
    ON_CALL(*this, get_v).WillByDefault(testing::ReturnRef(sigma_l_));
  }

  MOCK_METHOD(Matrix, predict, (ConstMatrixRef), (const override));
  MOCK_METHOD(Estimator &, consolidate_impl, (ConstMatrixRef, ConstMatrixRef, Requests), (override));
  MOCK_METHOD(double, score, (ConstMatrixRef, ConstMatrixRef), (const override));
  MOCK_METHOD(void, set, (Parameter, int), (override));
  MOCK_METHOD(void, set, (Parameter, double), (override));
  MOCK_METHOD(void, set, (Parameter, const Vector &), (override));
  [[nodiscard]] std::unique_ptr<Estimator> clone() const override {
    return std::make_unique<testing::NiceMock<MockEstimator_>>(predictions_, expected_sigma_l_, expected_sigma_f_,
                                                               expected_regularization_constant_, expected_degree_);
  }
  MOCK_METHOD(double, get_d, (Parameter), (const override));
  MOCK_METHOD(int, get_i, (Parameter), (const override));
  MOCK_METHOD(const Vector &, get_v, (Parameter), (const override));

  [[nodiscard]] double expected_sigma_f() const { return expected_sigma_f_; }
  [[nodiscard]] const Vector &expected_sigma_l() const { return expected_sigma_l_; }
  [[nodiscard]] double expected_regularization_constant() const { return expected_regularization_constant_; }
  [[nodiscard]] int expected_degree() const { return expected_degree_; }

 private:
  Matrix predictions_;
  Vector sigma_l_, expected_sigma_l_;
  double sigma_f_, expected_sigma_f_;
  double regularization_constant_, expected_regularization_constant_;
  int degree_, expected_degree_;
};

using MockEstimator = testing::NiceMock<MockEstimator_>;

class TestGridSearchTuner : public ::testing::TestWithParam<std::size_t> {
 protected:
  const int num_samples_{10};
  const int dim_{3};
  const Matrix training_inputs_{Matrix::Random(num_samples_, dim_)};
  const Matrix training_outputs_{Matrix::Random(num_samples_, 2)};
  // TODO(tend): This is needed when tuning with the feature map, because we need to match the sigma_l dimension (?)
  const Matrix training_outputs_full_{Matrix::Random(num_samples_, dim_)};
  const std::vector<ParameterValues> parameters_{
      ParameterValues{Parameter::SIGMA_L, std::vector<Vector>{Vector::Constant(dim_, 0.1), Vector::Constant(dim_, 1.0),
                                                              Vector::Constant(dim_, 10.0)}},
      ParameterValues{Parameter::SIGMA_F, 0.1, 1.0, 10.1},
      ParameterValues{Parameter::REGULARIZATION_CONSTANT, 1e-6, 1e-2, 10.0, 11.0},
      ParameterValues{Parameter::DEGREE, 1, 2}};
  const ParameterValues &sigma_l_values_{parameters_[0]};
  const ParameterValues &sigma_f_values_{parameters_[1]};
  const ParameterValues &regularization_constant_values_{parameters_[2]};
  const ParameterValues &degree_values_{parameters_[3]};
  const std::vector<Index> parameters_max_indices_{
      static_cast<Index>(parameters_[0].size()), static_cast<Index>(parameters_[1].size()),
      static_cast<Index>(parameters_[2].size()), static_cast<Index>(parameters_[3].size())};
  const int num_frequencies_{4};
  const RectSet X_bounds_{{-1, 1}, {-1, 1}, {-1, 1}};
  const int num_iterations_ = static_cast<int>(std::accumulate(
      parameters_max_indices_.begin(), parameters_max_indices_.end(), Index{1}, std::multiplies<Index>()));
  const GridSearchTuner tuner_{parameters_, GetParam()};
};

INSTANTIATE_TEST_SUITE_P(TestGridSearchTuner, TestGridSearchTuner, testing::Values(1, 2));

TEST_P(TestGridSearchTuner, Constructor) { EXPECT_NO_THROW(GridSearchTuner{parameters_}); }

TEST_P(TestGridSearchTuner, ConstructorEstimator) {
  const KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(3), 0,
                                       std::make_shared<GridSearchTuner>(parameters_)};
  EXPECT_NE(dynamic_cast<const GridSearchTuner *>(regressor.tuner().get()), nullptr);
}

TEST_P(TestGridSearchTuner, Tune) {
  for (lucid::IndexIterator it{parameters_max_indices_}; it; ++it) {
    testing::NiceMock<MockEstimator_> estimator{
        training_outputs_, sigma_l_values_.get<Vector>()[it[0]], sigma_f_values_.get<double>()[it[1]],
        regularization_constant_values_.get<double>()[it[2]], degree_values_.get<int>()[it[3]]};

    if (GetParam() == 1) {
      // Predict and score should be called once for each parameter combination
      EXPECT_CALL(estimator, predict).Times(num_iterations_);
      EXPECT_CALL(estimator, score).Times(num_iterations_);
      // Set should be called for each possible combination during the grid search + 1 to apply the best fit parameters
      EXPECT_CALL(estimator, consolidate_impl).Times(num_iterations_ + 1);
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<int>())).Times(num_iterations_ + 1);
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<double>())).Times(2 * (num_iterations_ + 1));
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<const Vector &>())).Times(num_iterations_ + 1);
    }

    estimator.fit(training_inputs_, training_outputs_, tuner_);

    ASSERT_EQ(estimator.get<Parameter::REGULARIZATION_CONSTANT>(), estimator.expected_regularization_constant());
    ASSERT_EQ(estimator.get<Parameter::SIGMA_F>(), estimator.expected_sigma_f());
    ASSERT_EQ(estimator.get<Parameter::DEGREE>(), estimator.expected_degree());
    ASSERT_EQ(estimator.get<Parameter::SIGMA_L>(), estimator.expected_sigma_l());
  }
}

TEST_P(TestGridSearchTuner, TuneOnline) {
  // Get the necessary information to iterate over all possible parameter combinations
  for (lucid::IndexIterator it{parameters_max_indices_}; it; ++it) {
    testing::NiceMock<MockEstimator_> estimator{
        training_outputs_, sigma_l_values_.get<Vector>()[it[0]], sigma_f_values_.get<double>()[it[1]],
        regularization_constant_values_.get<double>()[it[2]], degree_values_.get<int>()[it[3]]};

    if (GetParam() == 1) {
      // Predict and score should be called once for each parameter combination
      EXPECT_CALL(estimator, predict).Times(num_iterations_);
      EXPECT_CALL(estimator, score).Times(num_iterations_);
      // Set should be called for each possible combination during the grid search + 1 to apply the best fit parameters
      EXPECT_CALL(estimator, consolidate_impl).Times(num_iterations_ + 1);
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<int>())).Times(num_iterations_ + 1);
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<double>())).Times(2 * (num_iterations_ + 1));
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<const Vector &>())).Times(num_iterations_ + 1);
    }

    estimator.fit_online(
        training_inputs_, [this](const Estimator &, ConstMatrixRef) -> ConstMatrixRefCopy { return training_outputs_; },
        tuner_);

    // Even with multiple threads, the estimator should be set with the best parameters
    ASSERT_EQ(estimator.get<Parameter::REGULARIZATION_CONSTANT>(), estimator.expected_regularization_constant());
    ASSERT_EQ(estimator.get<Parameter::SIGMA_F>(), estimator.expected_sigma_f());
    ASSERT_EQ(estimator.get<Parameter::SIGMA_L>(), estimator.expected_sigma_l());
    ASSERT_EQ(estimator.get<Parameter::DEGREE>(), estimator.expected_degree());
  }
}

TEST_P(TestGridSearchTuner, TuneLinearTruncatedFourierFeatureMap) {
  for (lucid::IndexIterator it{parameters_max_indices_}; it; ++it) {
    LinearTruncatedFourierFeatureMap feature_map{num_frequencies_, sigma_l_values_.get<Vector>()[it[0]],
                                                 sigma_f_values_.get<double>()[it[1]], X_bounds_};
    testing::NiceMock<MockEstimator_> estimator{
        feature_map(training_outputs_full_), sigma_l_values_.get<Vector>()[it[0]], sigma_f_values_.get<double>()[it[1]],
        regularization_constant_values_.get<double>()[it[2]], degree_values_.get<int>()[it[3]]};

    if (GetParam() == 1) {
      // Predict and score should be called once for each parameter combination
      EXPECT_CALL(estimator, predict).Times(num_iterations_);
      EXPECT_CALL(estimator, score).Times(num_iterations_);
      // Set should be called for each possible combination during the grid search + 1 to apply the best fit parameters
      EXPECT_CALL(estimator, consolidate_impl).Times(num_iterations_ + 1);
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<int>())).Times(num_iterations_ + 1);
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<double>())).Times(2 * (num_iterations_ + 1));
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<const Vector &>())).Times(num_iterations_ + 1);
    }

    tuner_.tune<LinearTruncatedFourierFeatureMap>(estimator, training_inputs_, training_outputs_full_, num_frequencies_,
                                                  X_bounds_);

    // Even with multiple threads, the estimator should be set with the best parameters
    ASSERT_EQ(estimator.get<Parameter::REGULARIZATION_CONSTANT>(), estimator.expected_regularization_constant());
    ASSERT_EQ(estimator.get<Parameter::SIGMA_F>(), estimator.expected_sigma_f());
    ASSERT_EQ(estimator.get<Parameter::SIGMA_L>(), estimator.expected_sigma_l());
    ASSERT_EQ(estimator.get<Parameter::DEGREE>(), estimator.expected_degree());
  }
}
TEST_P(TestGridSearchTuner, TuneConstantTruncatedFourierFeatureMap) {
  for (lucid::IndexIterator it{parameters_max_indices_}; it; ++it) {
    ConstantTruncatedFourierFeatureMap feature_map{num_frequencies_, sigma_l_values_.get<Vector>()[it[0]],
                                                   sigma_f_values_.get<double>()[it[1]], X_bounds_};
    testing::NiceMock<MockEstimator_> estimator{
        feature_map(training_outputs_full_), sigma_l_values_.get<Vector>()[it[0]], sigma_f_values_.get<double>()[it[1]],
        regularization_constant_values_.get<double>()[it[2]], degree_values_.get<int>()[it[3]]};

    if (GetParam() == 1) {
      // Predict and score should be called once for each parameter combination
      EXPECT_CALL(estimator, predict).Times(num_iterations_);
      EXPECT_CALL(estimator, score).Times(num_iterations_);
      // Set should be called for each possible combination during the grid search + 1 to apply the best fit parameters
      EXPECT_CALL(estimator, consolidate_impl).Times(num_iterations_ + 1);
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<int>())).Times(num_iterations_ + 1);
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<double>())).Times(2 * (num_iterations_ + 1));
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<const Vector &>())).Times(num_iterations_ + 1);
    }

    tuner_.tune<ConstantTruncatedFourierFeatureMap>(estimator, training_inputs_, training_outputs_full_,
                                                    num_frequencies_, X_bounds_);

    // Even with multiple threads, the estimator should be set with the best parameters
    ASSERT_EQ(estimator.get<Parameter::REGULARIZATION_CONSTANT>(), estimator.expected_regularization_constant());
    ASSERT_EQ(estimator.get<Parameter::SIGMA_F>(), estimator.expected_sigma_f());
    ASSERT_EQ(estimator.get<Parameter::SIGMA_L>(), estimator.expected_sigma_l());
    ASSERT_EQ(estimator.get<Parameter::DEGREE>(), estimator.expected_degree());
  }
}
TEST_P(TestGridSearchTuner, TuneLogTruncatedFourierFeatureMap) {
  for (lucid::IndexIterator it{parameters_max_indices_}; it; ++it) {
    LogTruncatedFourierFeatureMap feature_map{num_frequencies_, sigma_l_values_.get<Vector>()[it[0]],
                                              sigma_f_values_.get<double>()[it[1]], X_bounds_};
    testing::NiceMock<MockEstimator_> estimator{
        feature_map(training_outputs_full_), sigma_l_values_.get<Vector>()[it[0]], sigma_f_values_.get<double>()[it[1]],
        regularization_constant_values_.get<double>()[it[2]], degree_values_.get<int>()[it[3]]};

    if (GetParam() == 1) {
      // Predict and score should be called once for each parameter combination
      EXPECT_CALL(estimator, predict).Times(num_iterations_);
      EXPECT_CALL(estimator, score).Times(num_iterations_);
      // Set should be called for each possible combination during the grid search + 1 to apply the best fit parameters
      EXPECT_CALL(estimator, consolidate_impl).Times(num_iterations_ + 1);
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<int>())).Times(num_iterations_ + 1);
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<double>())).Times(2 * (num_iterations_ + 1));
      EXPECT_CALL(estimator, set(testing::An<Parameter>(), testing::An<const Vector &>())).Times(num_iterations_ + 1);
    }

    tuner_.tune<LogTruncatedFourierFeatureMap>(estimator, training_inputs_, training_outputs_full_, num_frequencies_,
                                               X_bounds_);

    // Even with multiple threads, the estimator should be set with the best parameters
    ASSERT_EQ(estimator.get<Parameter::REGULARIZATION_CONSTANT>(), estimator.expected_regularization_constant());
    ASSERT_EQ(estimator.get<Parameter::SIGMA_F>(), estimator.expected_sigma_f());
    ASSERT_EQ(estimator.get<Parameter::SIGMA_L>(), estimator.expected_sigma_l());
    ASSERT_EQ(estimator.get<Parameter::DEGREE>(), estimator.expected_degree());
  }
}