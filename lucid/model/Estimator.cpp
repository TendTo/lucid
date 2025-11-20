/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Estimator.h"

#include <memory>
#include <ostream>
#include <string>

#include "lucid/model/Tuner.h"
#include "lucid/util/Stats.h"
#include "lucid/util/Timer.h"

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
Estimator& Estimator::consolidate(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs, Requests requests) {
  LUCID_TRACE_FMT("({}, {}, {})", LUCID_FORMAT_MATRIX(training_inputs), LUCID_FORMAT_MATRIX(training_outputs),
                  requests);
  TimerGuard tg{Stats::Scoped::top() ? &Stats::Scoped::top()->value().estimator_timer : nullptr};
  if (Stats::Scoped::top()) Stats::Scoped::top()->value().num_estimator_consolidations++;
  return consolidate_impl(training_inputs, training_outputs, requests);
}
Estimator& Estimator::fit_online(ConstMatrixRef training_inputs, const OutputComputer& training_outputs,
                                 const Tuner& tuner) {
  tuner.tune_online(*this, training_inputs, training_outputs);
  return *this;
}

std::string Estimator::to_string() const { return "Estimator( )"; }

std::ostream& operator<<(std::ostream& os, const Estimator& estimator) { return os << estimator.to_string(); }

}  // namespace lucid
