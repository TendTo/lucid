/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Optimiser class.
 */
#pragma once

#include <memory>

#include "lucid/tuning/Tuner.h"

namespace lucid::tuning {

class MedianHeuristicTuner final : public Tuner {
 public:
  explicit MedianHeuristicTuner(const Kernel& estimator);

 private:
  [[nodiscard]] Vector optimise_impl(const Matrix& kernel, const Matrix& y) const override;
};

}  // namespace lucid::tuning
// lucid
