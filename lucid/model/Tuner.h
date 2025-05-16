/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Tuner class.
 */
#pragma once

#include "lucid/model/Kernel.h"

namespace lucid {

// Forward declarations
class Estimator;

/**
 * Optimise the kernel hyperparameters.
 * Given a kernel, the optimiser finds the best hyperparameters for the kernel.
 * The optimiser subclass determines the optimisation algorithm.
 */
class Tuner {
 public:
  virtual ~Tuner() = default;

  /**
   * Optimise the kernel hyperparameters.
   * Starting from the initial guess, the optimiser finds the best hyperparameters for the kernel.
   * @param estimator estimator to optimise
   * @param training_inputs training input data. The number of rows should be equal to the number of training outputs
   * @param training_outputs training output data. The number of rows should be equal to the number of training inputs
   */
  void tune(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const;

 protected:
  /**
   * Optimise the kernel hyperparameters.
   * Starting from the initial guess, the optimiser finds the best hyperparameters for the kernel.
   * @param estimator estimator to optimise
   * @param training_inputs training input data. The number of rows should be equal to the number of training outputs
   * @param training_outputs training output data. The number of rows should be equal to the number of training inputs
   */
  virtual void tune_impl(Estimator& estimator, ConstMatrixRef training_inputs,
                         ConstMatrixRef training_outputs) const = 0;
};

}  // namespace lucid
