/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GradientOptimizable class.
 */
#pragma once

#include "lucid/lib/eigen.h"

namespace lucid {

/**
 * Interface for objects that can be optimised using gradient-based methods (e.g., LbfgsTuner).
 * It provides access to the objective value and the gradient of the objective function needed for optimisation.
 */
class GradientOptimizable {
 public:
  GradientOptimizable() = default;

  /** @getter{objective value, gradient optimizable object} */
  [[nodiscard]] double objective_value() const { return objective_value_; }
  /** @getter{gradient, gradient optimizable object} */
  [[nodiscard]] const Vector& gradient() const { return gradient_; }

 private:
  double objective_value_;  ///< Objective value of the optimisation problem
  Vector gradient_;         ///< Gradient of the objective function
};

}  // namespace lucid
