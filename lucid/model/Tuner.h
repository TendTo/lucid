/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Tuner class.
 */
#pragma once

#include <iosfwd>

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
  explicit Tuner() = default;
  Tuner(const Tuner &tuner) = default;
  Tuner(Tuner &&tuner) = default;
  Tuner &operator=(const Tuner &tuner) = default;
  Tuner &operator=(Tuner &&tuner) = default;
  virtual ~Tuner() = default;

  /**
   * Optimise the kernel hyperparameters.
   * Starting from the initial guess, the optimiser finds the best hyperparameters for the kernel.
   * @note The method must call the consolidate method of the estimator to ensure that the kernel is ready to be used.
   * @param estimator estimator to optimise
   * @param training_inputs training input data. The number of rows should be equal to the number of training outputs
   * @param training_outputs training output data. The number of rows should be equal to the number of training inputs
   */
  void tune(Estimator &estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const;

 protected:
  /**
   * Optimise the kernel hyperparameters.
   * Starting from the initial guess, the optimiser finds the best hyperparameters for the kernel.
   * It is up to the subclass to determine the optimisation algorithm used.
   * @note The method must call the consolidate method of the estimator to ensure that the kernel is ready to be used.
   * @param estimator estimator to optimise
   * @param training_inputs training input data. The number of rows should be equal to the number of training outputs
   * @param training_outputs training output data. The number of rows should be equal to the number of training inputs
   */
  virtual void tune_impl(Estimator &estimator, ConstMatrixRef training_inputs,
                         ConstMatrixRef training_outputs) const = 0;
};

std::ostream &operator<<(std::ostream &os, const Tuner &tuner);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Tuner)

#endif
