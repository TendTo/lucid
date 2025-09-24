/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Optimiser class.
 */
#include "lucid/verification/Optimiser.h"

#include <string>
#include <utility>

#include "lucid/util/Stats.h"
#include "lucid/util/error.h"

namespace lucid {

Optimiser::Optimiser(const int T, const double gamma, const double epsilon, const double b_norm, const double b_kappa,
                     const double sigma_f, const double C_coeff, std::string problem_log_file, std::string iis_log_file)
    : T_{T},
      gamma_{gamma},
      epsilon_{epsilon},
      b_norm_{b_norm},
      b_kappa_{b_kappa},
      sigma_f_{sigma_f},
      C_coeff_{C_coeff},
      problem_log_file_{std::move(problem_log_file)},
      iis_log_file_{std::move(iis_log_file)} {
  LUCID_CHECK_ARGUMENT_CMP(T, >, 0);
  LUCID_CHECK_ARGUMENT_EXPECTED(
      problem_log_file_.empty() || (problem_log_file_.ends_with(".lp") || problem_log_file_.ends_with(".mps") ||
                                    problem_log_file_.ends_with(".hxb")),
      "problem_log_file", problem_log_file_, "must be a valid file path with .lp, .mps or .hxb extension");
  LUCID_CHECK_ARGUMENT_EXPECTED(iis_log_file_.empty() || iis_log_file_.ends_with(".ilp"), "iis_log_file", iis_log_file_,
                                "must be a valid file path with .ilp extension");
}
Optimiser::Optimiser(std::string problem_log_file, std::string iis_log_file)
    : Optimiser(1, 1, 0, 1, 1, 1, 1, std::move(problem_log_file), std::move(iis_log_file)) {}

bool Optimiser::solve_fourier_barrier_synthesis(const FourierBarrierSynthesisParameters& params,
                                                const SolutionCallback& cb) const {
  TimerGuard tg{Stats::Scoped::top() ? &Stats::Scoped::top()->value().optimiser_timer : nullptr};
  return solve_fourier_barrier_synthesis_impl(params, cb);
}

}  // namespace lucid
