/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * MedianHeuristicTuner class.
 */
#pragma once

#include <memory>

#include "lucid/tuning/Tuner.h"

namespace lucid::tuning {

class MedianHeuristicTuner final : public Tuner {
  void tune_impl(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const override;
};

}  // namespace lucid::tuning
// lucid
