/**
 * @author TendTo
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/Scorer.h"

using lucid::ConstMatrixRef;
using lucid::Index;
using lucid::Matrix;
using lucid::Scalar;
using lucid::Vector;
using lucid::scorer::r2_score;
using lucid::scorer::Scorer;

class MockEstimator final : public lucid::Estimator {
 public:
  explicit MockEstimator(Matrix predictions) : predictions_(std::move(predictions)) {}

  [[nodiscard]] Matrix predict(ConstMatrixRef) const override { return predictions_; }
  [[nodiscard]] std::unique_ptr<lucid::Estimator> clone() const override {
    return std::make_unique<MockEstimator>(predictions_);
  }
  [[nodiscard]] bool has(const lucid::Parameter) const override { return false; }
  Estimator& consolidate(ConstMatrixRef, ConstMatrixRef) override { return *this; }
  [[nodiscard]] double score(ConstMatrixRef, ConstMatrixRef) const override { return 0.0; }

 private:
  Matrix predictions_;
};

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

TEST(TestScorer, R2ScoreMatchingSignature) { static_assert(std::is_convertible_v<decltype(r2_score), Scorer>); }