/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Scorer.h"

#include "lucid/util/error.h"

namespace lucid::scorer {

double r2_score(const Estimator& estimator, ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) {
  LUCID_CHECK_ARGUMENT_EXPECTED(evaluation_inputs.rows() > 1, "evaluation_inputs.rows()", evaluation_inputs.rows(),
                                "> 1");
  LUCID_CHECK_ARGUMENT_EXPECTED(evaluation_inputs.rows() == evaluation_outputs.rows(), "evaluation_inputs.rows()",
                                evaluation_inputs.rows(), evaluation_outputs.rows());
  const Matrix predictions = estimator.predict(evaluation_inputs);
  LUCID_ASSERT(predictions.rows() == evaluation_outputs.rows(),
               "The number of rows in predictions must match the number of rows in evaluation outputs.");
  LUCID_CHECK_ARGUMENT_EXPECTED(evaluation_outputs.cols() == predictions.cols(), "evaluation_outputs.cols()",
                                evaluation_outputs.cols(), predictions.cols());
  const double ss_res = (evaluation_outputs - predictions).array().square().sum();
  const double ss_tot = (evaluation_outputs.array() - evaluation_outputs.mean()).square().sum();
  return 1.0 - (ss_res / ss_tot);
}

double mse_score(const Estimator& estimator, ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) {
  LUCID_CHECK_ARGUMENT_EXPECTED(evaluation_inputs.rows() > 1, "evaluation_inputs.rows()", evaluation_inputs.rows(),
                                "> 1");
  LUCID_CHECK_ARGUMENT_EXPECTED(evaluation_inputs.rows() == evaluation_outputs.rows(), "evaluation_inputs.rows()",
                                evaluation_inputs.rows(), evaluation_outputs.rows());
  const Matrix predictions = estimator.predict(evaluation_inputs);
  LUCID_ASSERT(predictions.rows() == evaluation_outputs.rows(),
               "The number of rows in predictions must match the number of rows in evaluation outputs.");
  LUCID_CHECK_ARGUMENT_EXPECTED(evaluation_outputs.cols() == predictions.cols(), "evaluation_outputs.cols()",
                                evaluation_outputs.cols(), predictions.cols());
  const double mse = (evaluation_outputs - predictions).array().square().mean();
  LUCID_ASSERT(mse >= 0.0, "Mean squared error must be non-negative.");
  return -mse;
}

double rmse_score(const Estimator& estimator, ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) {
  LUCID_CHECK_ARGUMENT_EXPECTED(evaluation_inputs.rows() > 1, "evaluation_inputs.rows()", evaluation_inputs.rows(),
                                "> 1");
  LUCID_CHECK_ARGUMENT_EXPECTED(evaluation_inputs.rows() == evaluation_outputs.rows(), "evaluation_inputs.rows()",
                                evaluation_inputs.rows(), evaluation_outputs.rows());
  const Matrix predictions = estimator.predict(evaluation_inputs);
  LUCID_ASSERT(predictions.rows() == evaluation_outputs.rows(),
               "The number of rows in predictions must match the number of rows in evaluation outputs.");
  LUCID_CHECK_ARGUMENT_EXPECTED(evaluation_outputs.cols() == predictions.cols(), "evaluation_outputs.cols()",
                                evaluation_outputs.cols(), predictions.cols());
  const double rmse = std::sqrt((evaluation_outputs - predictions).array().square().mean());
  LUCID_ASSERT(rmse >= 0.0, "Root mean squared error must be non-negative.");
  return -rmse;
}

}  // namespace lucid::scorer
