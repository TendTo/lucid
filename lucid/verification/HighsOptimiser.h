/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * HighsOptimiser class.
 */
#pragma once

#include <iosfwd>
#include <map>
#include <string>

#include "lucid/lib/eigen.h"
#include "lucid/verification/Optimiser.h"

namespace lucid {

/**
 * Linear optimiser using the [HiGHS](https://ergo-code.github.io/HiGHS/dev/) mathematical library.
 */
class HighsOptimiser final : public Optimiser {
 public:
  using Optimiser::Optimiser;
  /**
   * Construct a new Highs Optimiser object.
   * @param options map of options to set in the HiGHS solver
   * @param problem_log_file file to log the problem to. If empty, no logging is done
   * @param iis_log_file file to log the irreducible infeasible set (IIS) to, if found. If empty, no logging is done
   */
  explicit HighsOptimiser(std::map<std::string, std::string> options, std::string problem_log_file = "",
                          std::string iis_log_file = "");

  [[nodiscard]] std::string to_string() const override;

 private:
  bool solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem& problem,
                                            const SolutionCallback& cb) const override;

  std::map<std::string, std::string> options_;  ///< Map of options to set in the HiGHS solver
};

std::ostream& operator<<(std::ostream& os, const HighsOptimiser& optimiser);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::HighsOptimiser)

#endif
