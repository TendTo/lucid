/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Tuner class.
 */
#pragma once

#include <iosfwd>

#include "lucid/model/Estimator.h"
#include "lucid/model/Kernel.h"

namespace lucid {

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
   * @pre The number of rows in the training inputs should be equal to the number of rows in the training outputs.
   * @param estimator estimator to optimise
   * @param training_inputs training input data
   * @param training_outputs training output data
   */
  void tune(Estimator &estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const;
  /**
   * Optimise the kernel hyperparameters.
   * Starting from the initial guess, the optimiser finds the best hyperparameters for the kernel.
   * @note The method must call the consolidate method of the estimator to ensure that the kernel is ready to be used.
   * @pre The number of rows in the training inputs should be equal to the number of rows in the training outputs.
   * @param estimator estimator to optimise
   * @param training_inputs training input data
   * @param training_outputs training output data. It uses an OutputComputer to compute the outputs when needed
   */
  void tune_online(Estimator &estimator, ConstMatrixRef training_inputs, const OutputComputer &training_outputs) const;

 protected:
  /**
   * Optimise the kernel hyperparameters.
   * Starting from the initial guess, the optimiser finds the best hyperparameters for the kernel.
   * It is up to the subclass to determine the optimisation algorithm used.
   * @note The method must call the consolidate method of the estimator to ensure that the kernel is ready to be used.
   * @pre The number of rows in the training inputs should be equal to the number of rows in the training outputs.
   * @todo If the OutputComputer becomes a bottleneck, consider having a separate method for when the outputs are
   * available from the get-go.
   * @param estimator estimator to optimise
   * @param training_inputs training input data
   * @param training_outputs training output data. It uses an OutputComputer to compute the outputs when needed
   */
  virtual void tune_impl(Estimator &estimator, ConstMatrixRef training_inputs,
                         const OutputComputer &training_outputs) const = 0;
};

std::ostream &operator<<(std::ostream &os, const Tuner &tuner);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Tuner)

#endif
