/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "MedianHeuristicTuner.h"

#include "lucid/lib/eigen.h"
#include "lucid/model/Estimator.h"
#include "lucid/util/logging.h"

namespace lucid {

namespace {

/**
 * Compute the median of a vector.
 * If the vector has an even number of elements, the median is the lower of the two middle elements.
 * @param d vector
 * @return median of the vector
 * @see https://stackoverflow.com/a/62698308/15153171
 */
Scalar median(Vector& d) {
  auto r{d.reshaped()};
  std::ranges::sort(r);
  return (r.size() & 1) == 0 ? r.segment((r.size() - 2) / 2, 2).minCoeff() : r(r.size() / 2);
}

}  // namespace

void MedianHeuristicTuner::tune_impl(Estimator& estimator, ConstMatrixRef training_inputs,
                                     ConstMatrixRef training_outputs) const {
  Vector dist = pdist(training_inputs);
  median(dist);
  estimator.set(Parameter::LENGTH_SCALE, dist);
  estimator.consolidate(training_inputs, training_outputs);
}

}  // namespace lucid
