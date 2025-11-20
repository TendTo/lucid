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

std::string FourierBarrierSynthesisProblem::to_string() const {
  return fmt::format(
      "FourierBarrierSynthesisProblem( num_constraints( {} ) fxn_lattice.rows( {} ) fxn_lattice.cols( {} ) "
      "dn_lattice.rows( {} ) dn_lattice.cols( {} ) x_include_mask.size( {} ) x_exclude_mask.size( {} ) "
      "x0_include_mask.size( {} ) x0_exclude_mask.size( {} ) xu_include_mask.size( {} ) xu_exclude_mask.size( {} ) "
      "T( {} ) gamma( {} ) eta_coeff( {} ) min_x0_coeff( {} ) diff_sx0_coeff( {} ) gamma_coeff( {} ) "
      "max_xu_coeff( {} ) diff_sxu_coeff( {} ) ebk( {} ) c_ebk_coeff( {} ) min_d_coeff( {} ) diff_d_sx_coeff( {} ) "
      "max_x_coeff( {} ) diff_sx_coeff( {} ) )",
      num_constraints, fxn_lattice.rows(), fxn_lattice.cols(), dn_lattice.rows(), dn_lattice.cols(),
      x_include_mask.size(), x_exclude_mask.size(), x0_include_mask.size(), x0_exclude_mask.size(),
      xu_include_mask.size(), xu_exclude_mask.size(), T, gamma, eta_coeff, min_x0_coeff, diff_sx0_coeff, gamma_coeff,
      max_xu_coeff, diff_sxu_coeff, ebk, c_ebk_coeff, min_d_coeff, diff_d_sx_coeff, max_x_coeff, diff_sx_coeff);
}

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
  if (Stats::Scoped::top()) {
    Stats::Scoped::top()->value().num_variables =
        problem.fxn_lattice.cols() + FourierBarrierSynthesisProblem::num_extra_vars;
    Stats::Scoped::top()->value().num_constraints = problem.num_constraints;
  }
  return solve_fourier_barrier_synthesis_impl(problem, cb);
}

std::string Optimiser::to_string() const {
  return fmt::format("Optimiser(problem_log_file( {} ) iis_log_file( {} ) )", problem_log_file_, iis_log_file_);
}

std::ostream& operator<<(std::ostream& os, const FourierBarrierSynthesisProblem& problem) {
  return os << problem.to_string();
}
std::ostream& operator<<(std::ostream& os, const Optimiser& optimiser) { return os << optimiser.to_string(); }

}  // namespace lucid
