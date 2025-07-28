/**
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GradientOptimizable class.
 */
#pragma once

#include <memory>

#include "lucid/lib/eigen.h"
#include "lucid/model/Estimator.h"

namespace lucid {

/**
 * Interface for objects that can be optimised using gradient-based methods (e.g., LbfgsTuner).
 * It provides access to the objective value and the gradient of the objective function needed for optimisation.
 * @todo Instead of being a subclass of Estimator,
 * we should introduce a mixin subclass that supports oll kinds of mixins
 */
class GradientOptimizable : public Estimator {
 public:
  explicit GradientOptimizable(const Parameters parameters = NoParameters,
                               const std::shared_ptr<const Tuner>& tuner = nullptr)
      : Estimator{parameters, tuner}, objective_value_{-std::numeric_limits<double>::infinity()}, gradient_{} {}

  /** @getter{objective value, this object, If it has not been computed\, it will be @f$ -\infty @f$.} */
  [[nodiscard]] double objective_value() const { return objective_value_; }
  /** @getter{gradient, this object, If it has not been computed\, it will be a vector of size 0.} */
  [[nodiscard]] const Vector& gradient() const { return gradient_; }

 protected:
  double objective_value_;  ///< Objective value of the optimisation problem
  Vector gradient_;         ///< Gradient of the objective function
};

}  // namespace lucid
