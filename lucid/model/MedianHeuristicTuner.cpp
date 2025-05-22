/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "MedianHeuristicTuner.h"

#include "lucid/lib/eigen.h"
#include "lucid/model/Estimator.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

void MedianHeuristicTuner::tune_impl(Estimator& estimator, ConstMatrixRef training_inputs,
                                     ConstMatrixRef training_outputs) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(training_inputs.rows() == training_outputs.rows(), "training_inputs.rows()",
                                training_inputs.rows(), training_outputs.rows());
  Vector new_sigma_l{training_inputs.cols()};
  for (Index i = 0; i < training_inputs.cols(); ++i) {
    // Compute the pdist between all inputs for each dimension individually
    Vector dist = pdist(training_inputs.col(i));
    // Assign the median of the distances to the new sigma_l
    new_sigma_l(i) = median(dist);
  }
  estimator.set(Parameter::LENGTH_SCALE, new_sigma_l);
  estimator.consolidate(training_inputs, training_outputs);
}

}  // namespace lucid
