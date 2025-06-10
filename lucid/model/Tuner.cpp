/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Tuner.h"

#include <memory>

#include "lucid/model/GridSearchTuner.h"
#include "lucid/model/MedianHeuristicTuner.h"

namespace lucid {

void Tuner::tune(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const {
  tune_impl(estimator, training_inputs, training_outputs);
}
std::ostream& operator<<(std::ostream& os, const Tuner& tuner) {
  if (const auto* casted = dynamic_cast<const MedianHeuristicTuner*>(&tuner)) return os << *casted;
  if (const auto* casted = dynamic_cast<const GridSearchTuner*>(&tuner)) return os << *casted;
  return os << "Tuner( )";
}

}  // namespace lucid
