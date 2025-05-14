/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Optimiser class.
 */
#pragma once

#include <memory>

#include "lucid/math/Kernel.h"
#include "lucid/math/Sampler.h"

namespace lucid::tuning {

/**
 * Optimise the kernel hyperparameters.
 * Given a kernel, the optimiser finds the best hyperparameters for the kernel.
 * The optimiser subclass determines the optimisation algorithm.
 */
class Optimiser {
 public:
  /** Construct a new Optimiser object. */
  explicit Optimiser(const Kernel& estimator);
  Optimiser(const Optimiser&) = default;
  Optimiser(Optimiser&&) = default;
  virtual ~Optimiser() = default;
  /**
   * Optimise the kernel hyperparameters.
   * Starting from the initial guess, the optimiser finds the best hyperparameters for the kernel.
   * @return optimised kernel
   */
  [[nodiscard]] Vector optimise(const Matrix& x, const Matrix& y) const;

 protected:
  /**
   * Optimise the kernel hyperparameters.
   * Starting from the initial guess, the optimiser finds the best hyperparameters for the kernel.
   * @return optimised kernel
   */
  [[nodiscard]] virtual Vector optimise_impl(const Matrix& x, const Matrix& y) const = 0;

  const Kernel& estimator_;  ///< Kernel to optimise.
};

}  // namespace lucid::tuning
