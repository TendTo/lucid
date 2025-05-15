/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/Estimator.h"

#include <ostream>

namespace lucid {

Estimator::Estimator(std::unique_ptr<tuning::Tuner>&& tuner) : tuner_{std::move(tuner)} {}
Matrix Estimator::operator()(ConstMatrixRef x) const { return predict(x); }
Estimator& Estimator::fit(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) {
  return fit(training_inputs, training_outputs, *tuner_);
}

std::ostream& operator<<(std::ostream& os, const Estimator&) { return os << "Estimator"; }

}  // namespace lucid
