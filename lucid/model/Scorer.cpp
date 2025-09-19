/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Scorer.h"

#include "lucid/util/error.h"

namespace lucid::scorer {

double r2_score(ConstMatrixRef x, ConstMatrixRef y) {
  LUCID_CHECK_ARGUMENT_CMP(y.rows(), >, 1);
  LUCID_CHECK_ARGUMENT_EQ(x.rows(), y.rows());
  LUCID_CHECK_ARGUMENT_EQ(x.cols(), y.cols());
  const double ss_res = (y - x).array().square().sum();
  const double ss_tot = (y.array() - y.mean()).square().sum();
  return 1.0 - (ss_res / ss_tot);
}
double r2_score(const Estimator& estimator, ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) {
  LUCID_CHECK_ARGUMENT_CMP(evaluation_inputs.rows(), >, 1);
  LUCID_CHECK_ARGUMENT_EQ(evaluation_inputs.rows(), evaluation_outputs.rows());
  const Matrix predictions = estimator.predict(evaluation_inputs);
  LUCID_CHECK_ARGUMENT_EQ(evaluation_outputs.cols(), predictions.cols());
  return r2_score(predictions, evaluation_outputs);
}

double mse_score(ConstMatrixRef x, ConstMatrixRef y) {
  LUCID_CHECK_ARGUMENT_CMP(y.rows(), >, 1);
  LUCID_CHECK_ARGUMENT_EQ(x.rows(), y.rows());
  LUCID_CHECK_ARGUMENT_EQ(x.cols(), y.cols());
  const double mse = (y - x).array().square().mean();
  LUCID_ASSERT(mse >= 0.0, "Mean squared error must be non-negative.");
  return -mse;  // Return negative to follow the convention of scorer functions
}

double mse_score(const Estimator& estimator, ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) {
  LUCID_CHECK_ARGUMENT_CMP(evaluation_inputs.rows(), >, 1);
  LUCID_CHECK_ARGUMENT_EQ(evaluation_inputs.rows(), evaluation_outputs.rows());
  const Matrix predictions = estimator.predict(evaluation_inputs);
  LUCID_CHECK_ARGUMENT_EQ(evaluation_outputs.cols(), predictions.cols());
  return mse_score(predictions, evaluation_outputs);
}

double rmse_score(ConstMatrixRef x, ConstMatrixRef y) {
  LUCID_CHECK_ARGUMENT_CMP(y.rows(), >, 1);
  LUCID_CHECK_ARGUMENT_EQ(x.rows(), y.rows());
  LUCID_CHECK_ARGUMENT_EQ(x.cols(), y.cols());
  const double rmse = std::sqrt((y - x).array().square().mean());
  LUCID_ASSERT(rmse >= 0.0, "Root mean squared error must be non-negative.");
  return -rmse;  // Return negative to follow the convention of scorer functions
}

double rmse_score(const Estimator& estimator, ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) {
  LUCID_CHECK_ARGUMENT_CMP(evaluation_inputs.rows(), >, 1);
  LUCID_CHECK_ARGUMENT_EQ(evaluation_inputs.rows(), evaluation_outputs.rows());
  const Matrix predictions = estimator.predict(evaluation_inputs);
  LUCID_CHECK_ARGUMENT_EQ(evaluation_outputs.cols(), predictions.cols());
  return rmse_score(predictions, evaluation_outputs);
}

double mape_score(ConstMatrixRef x, ConstMatrixRef y) {
  LUCID_CHECK_ARGUMENT_CMP(y.rows(), >, 1);
  LUCID_CHECK_ARGUMENT_EQ(x.rows(), y.rows());
  LUCID_CHECK_ARGUMENT_EQ(x.cols(), y.cols());
  const double mape = ((y - x).array() / y.array()).abs();
  return -mape.mean();  // Return negative to follow the convention of scorer functions
}

double mape_score(const Estimator& estimator, ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) {
  LUCID_CHECK_ARGUMENT_CMP(evaluation_inputs.rows(), >, 1);
  LUCID_CHECK_ARGUMENT_EQ(evaluation_inputs.rows(), evaluation_outputs.rows());
  const Matrix predictions = estimator.predict(evaluation_inputs);
  LUCID_CHECK_ARGUMENT_EQ(evaluation_outputs.cols(), predictions.cols());
  return mape_score(predictions, evaluation_outputs);
}

}  // namespace lucid::scorer
