/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/tuning/Tuner.h"

#include <memory>

#include "lucid/math/GaussianKernel.h"
#include "lucid/tuning/LbfgsTuner.h"

namespace lucid::tuning {

void Tuner::tune(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const {
  tune_impl(estimator, training_inputs, training_outputs);
}

}  // namespace lucid::tuning
