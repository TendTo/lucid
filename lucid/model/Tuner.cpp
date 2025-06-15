/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Tuner.h"

#include <memory>

#include "LbfgsTuner.h"
#include "lucid/model/GridSearchTuner.h"
#include "lucid/model/MedianHeuristicTuner.h"
#include "lucid/util/error.h"

namespace lucid {

void Tuner::tune(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const {
  LUCID_TRACE_FMT("({}, {}, {})", estimator, LUCID_FORMAT_MATRIX(training_inputs),
                  LUCID_FORMAT_MATRIX(training_outputs));
  LUCID_CHECK_ARGUMENT_EQ(training_inputs.rows(), training_outputs.rows());
  tune_impl(estimator, training_inputs,
            [training_outputs](const Estimator&, ConstMatrixRef) { return training_outputs; });
}
void Tuner::tune_online(Estimator& estimator, ConstMatrixRef training_inputs,
                        const OutputComputer& training_outputs) const {
  LUCID_TRACE_FMT("({}, {}, <lambda>)", estimator, LUCID_FORMAT_MATRIX(training_inputs));
  tune_impl(estimator, training_inputs, training_outputs);
}
std::ostream& operator<<(std::ostream& os, const Tuner& tuner) {
  if (const auto* casted = dynamic_cast<const MedianHeuristicTuner*>(&tuner)) return os << *casted;
  if (const auto* casted = dynamic_cast<const GridSearchTuner*>(&tuner)) return os << *casted;
  if (const auto* casted = dynamic_cast<const LbfgsTuner*>(&tuner)) return os << *casted;
  return os << "Tuner( )";
}

}  // namespace lucid
