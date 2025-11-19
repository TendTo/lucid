/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/Optimiser.h"

#include <string>
#include <utility>

#include "lucid/util/Stats.h"
#include "lucid/util/error.h"
#include "lucid/verification/AlglibOptimiser.h"
#include "lucid/verification/GurobiOptimiser.h"
#include "lucid/verification/HighsOptimiser.h"
#include "lucid/verification/SoplexOptimiser.h"

namespace lucid {

Optimiser::Optimiser(std::string problem_log_file, std::string iis_log_file)
    : problem_log_file_{std::move(problem_log_file)}, iis_log_file_{std::move(iis_log_file)} {
  LUCID_CHECK_ARGUMENT_EXPECTED(
      problem_log_file_.empty() || (problem_log_file_.ends_with(".lp") || problem_log_file_.ends_with(".mps")),
      "problem_log_file", problem_log_file_, "must be a valid file path with .lp or .mps extension");
  LUCID_CHECK_ARGUMENT_EXPECTED(iis_log_file_.empty() || iis_log_file_.ends_with(".ilp"), "iis_log_file", iis_log_file_,
                                "must be a valid file path with .ilp extension");
}

bool Optimiser::solve_fourier_barrier_synthesis(const FourierBarrierSynthesisProblem& problem,
                                                const SolutionCallback& cb) const {
  TimerGuard tg{Stats::Scoped::top() ? &Stats::Scoped::top()->value().optimiser_timer : nullptr};
  return solve_fourier_barrier_synthesis_impl(problem, cb);
}

std::string Optimiser::to_string() const {
  return fmt::format("Optimiser(problem_log_file( {} ) iis_log_file( {} ) )", problem_log_file_, iis_log_file_);
}

std::ostream& operator<<(std::ostream& os, const Optimiser& optimiser) { return os << optimiser.to_string(); }

}  // namespace lucid
