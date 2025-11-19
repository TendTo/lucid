/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * AlglibOptimiser class.
 */
#pragma once

#include <iosfwd>
#include <utility>

#include "lucid/lib/eigen.h"
#include "lucid/verification/Optimiser.h"

namespace lucid {

/**
 * Linear optimiser using the [Alglib](https://www.alglib.net/) mathematical library.
 */
class AlglibOptimiser final : public Optimiser {
 public:
  using Optimiser::Optimiser;

  [[nodiscard]] std::string to_string() const override;

 private:
  bool solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem& problem,
                                            const SolutionCallback& cb) const override;
};

std::ostream& operator<<(std::ostream& os, const AlglibOptimiser& optimiser);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::AlglibOptimiser)

#endif
