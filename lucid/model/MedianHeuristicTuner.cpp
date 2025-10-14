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
                                     const OutputComputer& training_outputs) const {
  LUCID_CHECK_ARGUMENT_CMP(training_inputs.rows(), >, 1);
  Vector new_sigma_l{training_inputs.cols()};
  for (Index i = 0; i < training_inputs.cols(); ++i) {
    // Compute the pdist between all inputs for each dimension individually
    Vector dist = pdist(training_inputs.col(i));
    // Assign the median of the distances to the new sigma_l
    new_sigma_l(i) = median(dist);
  }
  if (estimator.get<Parameter::SIGMA_L>().size() == 1) {
    if (!(new_sigma_l.array() == new_sigma_l(0)).all()) {
      LUCID_WARN(
          "The estimator has an isotropic length scale, but the median heuristic computed an anisotropic one. "
          "We suggest using an anisotropic length scale. The length scale will be set to the value of the first dim.");
    }
    estimator.set<Parameter::SIGMA_L>(Vector{new_sigma_l.head<1>()});
  } else {
    estimator.set<Parameter::SIGMA_L>(new_sigma_l);
  }
  estimator.consolidate(training_inputs, training_outputs(estimator, training_inputs));
}

std::ostream& operator<<(std::ostream& os, const MedianHeuristicTuner&) { return os << "MedianHeuristicTuner( )"; }

}  // namespace lucid
