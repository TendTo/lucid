/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Optimiser class.
 */
#pragma once

#include "lucid/tuning/Optimiser.h"

namespace lucid::tuning {

class MedianHeuristicOptimiser final : public Optimiser {
 public:
  explicit MedianHeuristicOptimiser(const Sampler& sampler, Dimension num_samples = 100);

 private:
  [[nodiscard]] std::unique_ptr<Kernel> optimise_impl(const Kernel& kernel) const override;
};

}  // namespace lucid::tuning
// lucid
