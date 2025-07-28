/**
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Estimator.h"

#include <memory>
#include <ostream>

#include "lucid/model/KernelRidgeRegressor.h"
#include "lucid/model/Tuner.h"

namespace lucid {

Estimator::Estimator(const Parameters parameters, const std::shared_ptr<const Tuner>& tuner)
    : Parametrizable{parameters}, tuner_{tuner} {}
Matrix Estimator::operator()(ConstMatrixRef x) const { return predict(x); }

Estimator& Estimator::fit(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) {
  return tuner_ ? fit(training_inputs, training_outputs, *tuner_) : consolidate(training_inputs, training_outputs);
}

Estimator& Estimator::fit(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs, const Tuner& tuner) {
  tuner.tune(*this, training_inputs, training_outputs);
  return *this;
}
Estimator& Estimator::fit_online(ConstMatrixRef training_inputs, const OutputComputer& training_outputs) {
  return tuner_ ? fit_online(training_inputs, training_outputs, *tuner_)
                : consolidate(training_inputs, training_outputs(*this, training_inputs), Request::_);
}
Estimator& Estimator::fit_online(ConstMatrixRef training_inputs, const OutputComputer& training_outputs,
                                 const Tuner& tuner) {
  tuner.tune_online(*this, training_inputs, training_outputs);
  return *this;
}

std::ostream& operator<<(std::ostream& os, const Estimator& estimator) {
  if (const auto* casted = dynamic_cast<const KernelRidgeRegressor*>(&estimator)) return os << *casted;
  return os << "Estimator( )";
}

}  // namespace lucid
