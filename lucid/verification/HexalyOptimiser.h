/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * HexalyOptimiser class.
 */
#pragma once

#include <utility>

#include "lucid/lib/eigen.h"
#include "lucid/verification/Optimiser.h"

namespace lucid {

/**
 * Linear optimiser using the [Hexaly](https://www.hexaly.com/) solver.
 */
class HexalyOptimiser final : public Optimiser {
 public:
  using Optimiser::Optimiser;

  bool solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisParameters& params,
                                            const SolutionCallback& cb) const override;
};

}  // namespace lucid
