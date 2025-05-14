/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Optimiser class.
 */
#pragma once

#include <memory>

#include "lucid/tuning/Optimiser.h"

namespace lucid::tuning {

class MedianHeuristicOptimiser final : public Optimiser {
 public:
  explicit MedianHeuristicOptimiser(const Kernel& estimator);

 private:
  [[nodiscard]] Vector optimise_impl(const Matrix& kernel, const Matrix& y) const override;
};

}  // namespace lucid::tuning
// lucid
