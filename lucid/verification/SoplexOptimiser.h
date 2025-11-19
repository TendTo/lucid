/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * SoplexOptimiser class.
 */
#pragma once

#include <iosfwd>

#include "lucid/verification/Optimiser.h"

namespace lucid {

/**
 * Linear optimiser using the [SoPlex](https://soplex.zib.de/) solver.
 */
class SoplexOptimiser final : public Optimiser {
 public:
  using Optimiser::Optimiser;

  [[nodiscard]] std::string to_string() const override;

 private:
  bool solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem& problem,
                                            const SolutionCallback& cb) const override;
};

std::ostream& operator<<(std::ostream& os, const SoplexOptimiser& optimiser);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::SoplexOptimiser)

#endif
