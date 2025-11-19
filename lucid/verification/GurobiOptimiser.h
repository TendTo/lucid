/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GurobiOptimiser class.
 */
#pragma once

#include <iosfwd>
#include <utility>

#include "lucid/lib/eigen.h"
#include "lucid/verification/Optimiser.h"

namespace lucid {

/**
 * Linear optimiser using the [Gurobi](https://www.gurobi.com/) solver.
 */
class GurobiOptimiser final : public Optimiser {
 public:
  using Optimiser::Optimiser;

  [[nodiscard]] bool solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem& problem,
                                                          const SolutionCallback& cb) const override;

  [[nodiscard]] std::string to_string() const override;
};

std::ostream& operator<<(std::ostream& os, const GurobiOptimiser& optimiser);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::GurobiOptimiser)

#endif
