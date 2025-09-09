/**
 * @author TendTo
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "lucid/model/Scorer.h"

using lucid::ConstMatrixRef;
using lucid::Estimator;
using lucid::Index;
using lucid::Matrix;
using lucid::Parameter;
using lucid::Requests;
using lucid::Scalar;
using lucid::Vector;
using lucid::scorer::mse_score;
using lucid::scorer::r2_score;
using lucid::scorer::rmse_score;
using lucid::scorer::Scorer;

class MockEstimator_ : public Estimator {
 public:
  explicit MockEstimator_(Matrix predictions) : predictions_{std::move(predictions)} {
    ON_CALL(*this, predict).WillByDefault(testing::Return(predictions_));
    ON_CALL(*this, consolidate_impl).WillByDefault(testing::ReturnRef(*this));
  }

  MOCK_METHOD(Matrix, predict, (ConstMatrixRef), (const override));
  MOCK_METHOD(bool, has, (lucid::Parameter), (const override));
  MOCK_METHOD(Estimator&, consolidate_impl, (ConstMatrixRef, ConstMatrixRef, Requests), (override));
  MOCK_METHOD(double, score, (ConstMatrixRef, ConstMatrixRef), (const override));
  [[nodiscard]] std::unique_ptr<Estimator> clone() const override {
    return std::make_unique<MockEstimator_>(predictions_);
  }

 private:
  Matrix predictions_;
};

using MockEstimator = testing::NiceMock<MockEstimator_>;

TEST(TestScorer, R2ScorePerfectPredictions) {
  const Matrix inputs{Matrix::Random(20, 5)};
  const Matrix outputs{Matrix::Random(20, 2)};
  const MockEstimator estimator{outputs};

  EXPECT_DOUBLE_EQ(r2_score(estimator, inputs, outputs), 1.0);
}

TEST(TestScorer, R2ScoreMeanPredictions) {
  const Matrix inputs{Matrix::Random(30, 2)};
  const Matrix outputs{Matrix::Random(30, 1)};

  const Matrix mean_predictions{Matrix::Constant(outputs.rows(), outputs.cols(), outputs.mean())};
  const MockEstimator estimator{mean_predictions};

  EXPECT_DOUBLE_EQ(r2_score(estimator, inputs, outputs), 0.0);
}

TEST(TestScorer, R2ScoreWorseThanMean) {
  const Matrix inputs{Matrix::Random(15, 3)};
  const Matrix outputs{Matrix::Random(15, 1)};

  const Matrix bad_predictions = -2.0 * outputs.array() + outputs.mean();
  const MockEstimator estimator{bad_predictions};

  EXPECT_LT(r2_score(estimator, inputs, outputs), 0.0);
}

TEST(TestScorer, R2ScoreZeroVarianceOutputs) {
  const Matrix inputs{Matrix::Random(10, 2)};
  const Matrix outputs{Matrix::Constant(10, 1, 5.0)};

  const Matrix perfect_predictions{Matrix::Constant(10, 1, 5.0)};
  const MockEstimator perfectEstimator{perfect_predictions};

  // Since there's no variance in the outputs, R² score is undefined (division by zero)
  // We'll check that it doesn't crash and returns a reasonable value
  const double score = r2_score(perfectEstimator, inputs, outputs);
  EXPECT_TRUE(std::isnan(score));
}

TEST(TestScorer, R2ScoreMultidimensionalOutputs) {
  const Matrix inputs{Matrix::Random(15, 3)};
  const Matrix outputs{Matrix::Random(15, 15)};

  // Create predictions with varying accuracy
  Matrix predictions = outputs;
  predictions.col(0) = outputs.col(0) * 0.9;  // First dimension less accurate
  const MockEstimator estimator{predictions};

  const double score = r2_score(estimator, inputs, outputs);
  EXPECT_LT(score, 1.0);
  EXPECT_GT(score, 0.0);
}

TEST(TestScorer, R2ScoreLargeValues) {
  const Matrix inputs{Matrix::Random(10, 2)};
  const Matrix outputs = 1e6 * Matrix::Random(10, 1);  // Very large values

  // Almost perfect predictions, but with small errors
  const Matrix predictions = outputs + Matrix::Constant(10, 1, 1.0);
  const MockEstimator estimator{predictions};

  const double score = r2_score(estimator, inputs, outputs);
  EXPECT_LT(score, 1.0);
  EXPECT_GT(score, 0.9);  // Should still be good
}

TEST(TestScorer, R2ScoreSmallestDataset) {
  const Matrix inputs{Matrix::Random(2, 1)};
  const Matrix outputs{Matrix::Random(2, 1)};

  const MockEstimator perfectEstimator{outputs};
  EXPECT_DOUBLE_EQ(r2_score(perfectEstimator, inputs, outputs), 1.0);

  const Matrix mean_predictions{Matrix::Constant(2, 1, outputs.mean())};
  const MockEstimator meanEstimator{mean_predictions};
  EXPECT_DOUBLE_EQ(r2_score(meanEstimator, inputs, outputs), 0.0);
}

TEST(TestScorer, R2ScoreSingleDataSample) {
  const Matrix inputs{Matrix::Random(1, 5)};
  const Matrix outputs{Matrix::Random(1, 6)};

  const MockEstimator perfectEstimator{outputs};
  EXPECT_THROW(r2_score(perfectEstimator, inputs, outputs), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestScorer, R2ScoreLinearFunction) {
  // Test with a known function: y = 2x + 1
  constexpr int n_samples = 10;
  const Matrix inputs{Matrix::Random(n_samples, 1)};
  Matrix outputs(n_samples, 1);

  // Generate outputs based on y = 2x + 1
  for (int i = 0; i < n_samples; i++) {
    outputs(i, 0) = 2 * inputs(i, 0) + 1;
  }

  // Predictions from imperfect linear model: y = 1.9x + 0.9
  Matrix predictions(n_samples, 1);
  for (int i = 0; i < n_samples; i++) {
    predictions(i, 0) = 1.9 * inputs(i, 0) + 0.9;
  }

  const MockEstimator estimator{predictions};
  const double score = r2_score(estimator, inputs, outputs);

  // Should be a good but not perfect fit
  EXPECT_LT(score, 1.0);
  EXPECT_GT(score, 0.9);
}

TEST(TestScorer, R2ScoreMismatchedDimensions) {
  const Matrix inputs{Matrix::Random(10, 3)};
  const Matrix outputs{Matrix::Random(10, 2)};

  // Predictions with wrong number of columns
  const Matrix wrong_predictions{Matrix::Random(10, 1)};
  const MockEstimator estimator{wrong_predictions};

  EXPECT_THROW(r2_score(estimator, inputs, outputs), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestScorer, R2ScoreDirectPrediction) {
  const Matrix inputs{Matrix::Random(10, 3)};
  const Matrix outputs{Matrix::Random(10, 2)};
  const MockEstimator estimator{Matrix::Random(10, 2)};

  EXPECT_EQ(r2_score(estimator, inputs, outputs), r2_score(estimator(inputs), outputs));
}

TEST(TestScorer, MSEScorePerfectPredictions) {
  const Matrix inputs{Matrix::Random(20, 5)};
  const Matrix outputs{Matrix::Random(20, 2)};
  const MockEstimator estimator{outputs};

  EXPECT_DOUBLE_EQ(mse_score(estimator, inputs, outputs), 0.0);
}

TEST(TestScorer, MSEScoreImperfectPredictions) {
  const Matrix inputs{Matrix::Random(20, 5)};
  const Matrix outputs{Matrix::Random(20, 2)};

  // Add constant error of 1.0 to all predictions
  const Matrix predictions = outputs.array() + 1.0;
  const MockEstimator estimator{predictions};

  // Expected MSE is 1.0 since all errors are exactly 1.0
  EXPECT_DOUBLE_EQ(mse_score(estimator, inputs, outputs), -1.0);
}

TEST(TestScorer, MSEScoreVaryingErrors) {
  const Matrix inputs{Matrix::Random(10, 3)};
  const Matrix outputs{Matrix::Random(10, 2)};

  // Create predictions with known errors: 1st column has error 1.0, 2nd column has error 2.0
  Matrix predictions = outputs;
  predictions.col(0).array() += 1.0;
  predictions.col(1).array() += 2.0;
  const MockEstimator estimator{predictions};

  // Expected MSE is average of squared errors: (1^2 + 2^2)/2 = 2.5
  EXPECT_DOUBLE_EQ(mse_score(estimator, inputs, outputs), -2.5);
}

TEST(TestScorer, MSEScoreMultidimensionalOutputs) {
  const Matrix inputs{Matrix::Random(15, 3)};
  const Matrix outputs{Matrix::Random(15, 5)};

  // Create predictions with different errors for different dimensions
  Matrix predictions = outputs;
  for (int i = 0; i < outputs.cols(); ++i) {
    predictions.col(i).array() += static_cast<double>(i);  // Error increases with column index
  }
  const MockEstimator estimator{predictions};

  // Expected MSE = (0^2 + 1^2 + 2^2 + 3^2 + 4^2)/5 = 6.0
  EXPECT_DOUBLE_EQ(mse_score(estimator, inputs, outputs), -6.0);
}

TEST(TestScorer, MSEScoreLargeValues) {
  const Matrix inputs{Matrix::Random(10, 2)};
  const Matrix outputs = 1e6 * Matrix::Random(10, 1);  // Very large values

  // Add small errors of 10.0
  const Matrix predictions = outputs.array() + 10.0;
  const MockEstimator estimator{predictions};

  // Expected MSE = 10^2 = 100
  EXPECT_DOUBLE_EQ(mse_score(estimator, inputs, outputs), -100.0);
}

TEST(TestScorer, MSEScoreSmallValues) {
  const Matrix inputs{Matrix::Random(10, 2)};
  const Matrix outputs = 1e-6 * Matrix::Random(10, 1);  // Very small values

  // Add small errors of 1e-6
  const Matrix predictions = outputs.array() + 1e-6;
  const MockEstimator estimator{predictions};

  // Expected MSE = (1e-6)^2 = 1e-12
  EXPECT_DOUBLE_EQ(mse_score(estimator, inputs, outputs), -1e-12);
}

TEST(TestScorer, MSEScoreSmallestDataset) {
  const Matrix inputs{Matrix::Random(2, 1)};
  const Matrix outputs{Matrix::Random(2, 1)};

  // Perfect predictions
  const MockEstimator perfectEstimator{outputs};
  EXPECT_DOUBLE_EQ(mse_score(perfectEstimator, inputs, outputs), 0.0);

  // Imperfect predictions with known error
  Matrix predictions = outputs;
  predictions.array() += 2.0;
  const MockEstimator imperfectEstimator{predictions};
  EXPECT_DOUBLE_EQ(mse_score(imperfectEstimator, inputs, outputs), -4.0);  // 2^2 = 4
}

TEST(TestScorer, MSEScoreSingleDataSample) {
  const Matrix inputs{Matrix::Random(1, 5)};
  const Matrix outputs{Matrix::Random(1, 6)};

  const MockEstimator estimator{outputs};
  EXPECT_THROW(mse_score(estimator, inputs, outputs), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestScorer, MSEScoreLinearFunction) {
  // Test with a known function: y = 2x + 1
  constexpr int n_samples = 10;
  const Matrix inputs{Matrix::Random(n_samples, 1)};
  Matrix outputs(n_samples, 1);

  // Generate outputs based on y = 2x + 1
  for (int i = 0; i < n_samples; i++) {
    outputs(i, 0) = 2 * inputs(i, 0) + 1;
  }

  // Predictions from imperfect linear model: y = 2x + 2
  Matrix predictions(n_samples, 1);
  for (int i = 0; i < n_samples; i++) {
    predictions(i, 0) = 2 * inputs(i, 0) + 2;
  }

  const MockEstimator estimator{predictions};
  // Error is consistently 1.0, so MSE should be 1.0
  EXPECT_DOUBLE_EQ(mse_score(estimator, inputs, outputs), -1.0);
}

TEST(TestScorer, MSEScoreMismatchedDimensions) {
  const Matrix inputs{Matrix::Random(10, 3)};
  const Matrix outputs{Matrix::Random(10, 2)};

  // Predictions with wrong number of columns
  const Matrix wrong_predictions{Matrix::Random(10, 1)};
  const MockEstimator estimator{wrong_predictions};

  EXPECT_THROW(mse_score(estimator, inputs, outputs), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestScorer, MSEScoreMismatchedRows) {
  const Matrix inputs{Matrix::Random(10, 3)};
  const Matrix outputs{Matrix::Random(8, 2)};  // Different number of rows

  const Matrix predictions{Matrix::Random(8, 2)};
  const MockEstimator estimator{predictions};

  EXPECT_THROW(mse_score(estimator, inputs, outputs), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestScorer, MSEScoreAllNaN) {
  const Matrix inputs{Matrix::Random(5, 2)};
  Matrix outputs{Matrix::Random(5, 2)};
  outputs.setConstant(std::numeric_limits<double>::quiet_NaN());

  Matrix predictions{Matrix::Random(5, 2)};
  predictions.setConstant(std::numeric_limits<double>::quiet_NaN());
  const MockEstimator estimator{predictions};

#ifndef NDEBUG
  EXPECT_THROW(mse_score(estimator, inputs, outputs), lucid::exception::LucidAssertionException);
#else
  EXPECT_TRUE(std::isnan(mse_score(estimator, inputs, outputs)));
#endif
}

TEST(TestScorer, MSEScoreNegativePredictions) {
  const Matrix inputs{Matrix::Random(10, 2)};
  const Matrix outputs{Matrix::Constant(10, 1, 5.0)};

  // Predictions that are all negative
  const Matrix predictions{Matrix::Constant(10, 1, -5.0)};
  const MockEstimator estimator{predictions};

  // MSE = (5 - (-5))^2 = 10^2 = 100
  EXPECT_DOUBLE_EQ(mse_score(estimator, inputs, outputs), -100.0);
}

TEST(TestScorer, MSEScoreDirectPrediction) {
  const Matrix inputs{Matrix::Random(10, 3)};
  const Matrix outputs{Matrix::Random(10, 2)};
  const MockEstimator estimator{Matrix::Random(10, 2)};

  EXPECT_EQ(mse_score(estimator, inputs, outputs), mse_score(estimator(inputs), outputs));
}

TEST(TestScorer, RMSEScorePerfectPredictions) {
  const Matrix inputs{Matrix::Random(20, 3)};
  const Matrix outputs{Matrix::Random(20, 2)};
  const MockEstimator estimator{outputs};

  EXPECT_DOUBLE_EQ(rmse_score(estimator, inputs, outputs), 0.0);
}

TEST(TestScorer, RMSEScoreConstantError) {
  const Matrix inputs{Matrix::Random(10, 2)};
  const Matrix outputs{Matrix::Random(10, 1)};
  const Matrix predictions{outputs.array() + 3.0};
  const MockEstimator estimator{predictions};

  // RMSE = sqrt(mean((3)^2)) = 3, but score is -RMSE
  EXPECT_DOUBLE_EQ(rmse_score(estimator, inputs, outputs), -3.0);
}

TEST(TestScorer, RMSEScoreVaryingErrors) {
  const Matrix inputs{Matrix::Random(5, 2)};
  Matrix outputs{5, 2};
  outputs << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
  Matrix predictions{outputs};
  predictions(0, 0) += 1.0;
  predictions(1, 1) += 2.0;
  predictions(2, 0) -= 1.0;
  predictions(3, 1) -= 2.0;
  // RMSE = sqrt(mean([1^2, 2^2, 1^2, 2^2, 0, 0, 0, 0, 0, 0])) = sqrt(10/10) = 1
  const MockEstimator estimator{predictions};
  EXPECT_DOUBLE_EQ(rmse_score(estimator, inputs, outputs), -1.0);
}

TEST(TestScorer, RMSEScoreMultidimensionalOutputs) {
  const Matrix inputs{Matrix::Random(8, 3)};
  Matrix outputs{Matrix::Random(8, 4)};
  Matrix predictions{outputs};
  predictions.col(0).array() += 1.0;
  predictions.col(1).array() += 2.0;
  predictions.col(2).array() += 3.0;
  predictions.col(3).array() += 4.0;
  // RMSE = sqrt(mean([1^2, 2^2, 3^2, 4^2])) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386127875258306
  const MockEstimator estimator{predictions};
  EXPECT_NEAR(rmse_score(estimator, inputs, outputs), -2.7386127875258306, 1e-8);
}

TEST(TestScorer, RMSEScoreLargeValues) {
  const Matrix inputs{Matrix::Random(5, 2)};
  const Matrix outputs{1e6 * Matrix::Random(5, 1)};
  const Matrix predictions{outputs.array() + 1000.0};
  const MockEstimator estimator{predictions};

  EXPECT_DOUBLE_EQ(rmse_score(estimator, inputs, outputs), -1000.0);
}

TEST(TestScorer, RMSEScoreSmallValues) {
  const Matrix inputs{Matrix::Random(5, 2)};
  const Matrix outputs{1e-6 * Matrix::Random(5, 1)};
  const Matrix predictions{outputs.array() + 1e-6};
  const MockEstimator estimator{predictions};

  EXPECT_DOUBLE_EQ(rmse_score(estimator, inputs, outputs), -1e-6);
}

TEST(TestScorer, RMSEScoreSmallestDataset) {
  const Matrix inputs{Matrix::Random(2, 1)};
  const Matrix outputs{Matrix::Random(2, 1)};

  // Perfect predictions
  const MockEstimator perfectEstimator{outputs};
  EXPECT_DOUBLE_EQ(rmse_score(perfectEstimator, inputs, outputs), 0.0);

  // Imperfect predictions with known error
  Matrix predictions{outputs};
  predictions.array() += 2.0;
  const MockEstimator imperfectEstimator{predictions};
  EXPECT_DOUBLE_EQ(rmse_score(imperfectEstimator, inputs, outputs), -2.0);
}

TEST(TestScorer, RMSEScoreSingleDataSample) {
  const Matrix inputs{Matrix::Random(1, 3)};
  const Matrix outputs{Matrix::Random(1, 2)};
  const MockEstimator estimator{outputs};

  EXPECT_THROW(rmse_score(estimator, inputs, outputs), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestScorer, RMSEScoreMismatchedRows) {
  const Matrix inputs{Matrix::Random(10, 2)};
  const Matrix outputs{Matrix::Random(8, 2)};
  const Matrix predictions{Matrix::Random(8, 2)};
  const MockEstimator estimator{predictions};

  EXPECT_THROW(rmse_score(estimator, inputs, outputs), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestScorer, RMSEScoreMismatchedDimensions) {
  const Matrix inputs{Matrix::Random(10, 2)};
  const Matrix outputs{Matrix::Random(10, 3)};
  const Matrix predictions{Matrix::Random(10, 2)};
  const MockEstimator estimator{predictions};

  EXPECT_THROW(rmse_score(estimator, inputs, outputs), lucid::exception::LucidInvalidArgumentException);
}

TEST(TestScorer, RMSEScoreAllNaN) {
  const Matrix inputs{Matrix::Random(5, 2)};
  Matrix outputs{Matrix::Random(5, 2)};
  outputs.setConstant(std::numeric_limits<double>::quiet_NaN());
  Matrix predictions{Matrix::Random(5, 2)};
  predictions.setConstant(std::numeric_limits<double>::quiet_NaN());
  const MockEstimator estimator{predictions};

#ifndef NDEBUG
  EXPECT_THROW(rmse_score(estimator, inputs, outputs), lucid::exception::LucidAssertionException);
#else
  EXPECT_TRUE(std::isnan(rmse_score(estimator, inputs, outputs)));
#endif
}

TEST(TestScorer, RMSEScoreNegativePredictions) {
  const Matrix inputs{Matrix::Random(10, 2)};
  const Matrix outputs{Matrix::Constant(10, 1, 5.0)};
  const Matrix predictions{Matrix::Constant(10, 1, -5.0)};
  const MockEstimator estimator{predictions};

  // RMSE = sqrt(mean((5-(-5))^2)) = sqrt(100) = 10
  EXPECT_DOUBLE_EQ(rmse_score(estimator, inputs, outputs), -10.0);
}

TEST(TestScorer, RMSEScoreLinearFunction) {
  constexpr int n_samples = 10;
  const Matrix inputs{Matrix::Random(n_samples, 1)};
  Matrix outputs{n_samples, 1};
  for (int i = 0; i < n_samples; ++i) {
    outputs(i, 0) = 2 * inputs(i, 0) + 1;
  }
  Matrix predictions{n_samples, 1};
  for (int i = 0; i < n_samples; ++i) {
    predictions(i, 0) = 2 * inputs(i, 0) + 2;
  }
  const MockEstimator estimator{predictions};
  // Error is consistently 1.0, so RMSE should be 1.0
  EXPECT_DOUBLE_EQ(rmse_score(estimator, inputs, outputs), -1.0);
}

TEST(TestScorer, RMSEScoreDirectPrediction) {
  const Matrix inputs{Matrix::Random(10, 3)};
  const Matrix outputs{Matrix::Random(10, 2)};
  const MockEstimator estimator{Matrix::Random(10, 2)};

  EXPECT_EQ(rmse_score(estimator, inputs, outputs), rmse_score(estimator(inputs), outputs));
}
