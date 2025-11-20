/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Tuner.h"

#include <memory>
#include <string>

#include "lucid/util/ScopedValue.h"
#include "lucid/util/Stats.h"
#include "lucid/util/Timer.h"
#include "lucid/util/error.h"

namespace lucid {

void Tuner::tune(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const {
  LUCID_CHECK_ARGUMENT_EQ(training_inputs.rows(), training_outputs.rows());
  tune_online(estimator, training_inputs,
              [training_outputs](const Estimator&, ConstMatrixRef) { return training_outputs; });
}
void Tuner::tune_online(Estimator& estimator, ConstMatrixRef training_inputs,
                        const OutputComputer& training_outputs) const {
  LUCID_TRACE_FMT("({}, {}, training_outputs)", estimator, LUCID_FORMAT_MATRIX(training_inputs));
  TimerGuard tg{Stats::Scoped::top() ? &Stats::Scoped::top()->value().tuning_timer : nullptr};
  if (Stats::Scoped::top()) Stats::Scoped::top()->value().num_tuning++;
  tune_impl(estimator, training_inputs, training_outputs);
}

std::string Tuner::to_string() const { return "Tuner( )"; }

std::ostream& operator<<(std::ostream& os, const Tuner& tuner) { return os << tuner.to_string(); }

}  // namespace lucid
