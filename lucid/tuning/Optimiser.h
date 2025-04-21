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
  /**
   * Construct a new Optimiser object.
   * The sampler will be prompted to generate `num_samples` samples.
   * @param sampler sampling function
   * @param num_samples number of samples to generate
   */
  explicit Optimiser(const Sampler& sampler, Dimension num_samples = 100);
  Optimiser(const Optimiser&) = default;
  Optimiser(Optimiser&&) = default;
  virtual ~Optimiser() = default;
  /**
   * Optimise the kernel hyperparameters.
   * Starting from the initial guess, the optimiser finds the best hyperparameters for the kernel.
   * @param kernel initial guess for the optimisation
   * @return optimised kernel
   */
  [[nodiscard]] std::unique_ptr<Kernel> optimise(const Kernel& kernel) const;
  /**
   * Evaluate the performance of the kernel.
   * The metric used depends on the optimisation algorithm.
   * @param kernel kernel to evaluate
   * @return performance metric
   */
  [[nodiscard]] Scalar evaluate(const Kernel& kernel) const;
  /** @getter{number of samples to use in the optimisation, optimiser} */
  [[nodiscard]] Dimension num_samples() const { return num_samples_; }

 protected:
  /**
   * Optimise the kernel hyperparameters.
   * Starting from the initial guess, the optimiser finds the best hyperparameters for the kernel.
   * @param kernel initial guess for the optimisation
   * @return optimised kernel
   */
  [[nodiscard]] virtual std::unique_ptr<Kernel> optimise_impl(const Kernel& kernel) const = 0;

  const Dimension num_samples_;  ///< Number of samples to generate for the optimisation
  const Sampler& sampler_;       ///< Sampling function
};

}  // namespace lucid::tuning
