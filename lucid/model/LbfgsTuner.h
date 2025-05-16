/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * LbfgsTuner class.
 */
#pragma once

#include <memory>

#include "lucid/model/Kernel.h"
#include "lucid/model/Tuner.h"

namespace lucid {

/**
 * Optimiser that uses the L-BFGS algorithm.
 */
class LbfgsTuner final : public Tuner {
  void tune_impl(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const override;
};

}  // namespace lucid
