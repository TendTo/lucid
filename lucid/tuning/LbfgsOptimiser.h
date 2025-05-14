/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * LbfgsOptimiser class.
 */
#pragma once

#include <memory>

#include "lucid/math/Kernel.h"
#include "lucid/tuning/Optimiser.h"

namespace lucid::tuning {

/**
 * Optimiser that uses the L-BFGS algorithm.
 */
class LbfgsOptimiser final : public Optimiser {
 public:
  explicit LbfgsOptimiser(const Kernel& estimator);

 private:
  [[nodiscard]] Vector optimise_impl(const Matrix& x, const Matrix& y) const override;
};

}  // namespace lucid::tuning
