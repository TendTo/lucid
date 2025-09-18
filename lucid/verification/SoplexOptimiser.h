/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * SoplexOptimiser class.
 */
#pragma once

#include "lucid/verification/Optimiser.h"

namespace lucid {

/**
 * Linear optimiser using the [SoPlex](https://soplex.zib.de/) solver.
 */
class SoplexOptimiser final : public Optimiser {
 public:
  using Optimiser::Optimiser;

 private:
  bool solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisParameters& params,
                                            const SolutionCallback& cb) const override;
};

}  // namespace lucid
