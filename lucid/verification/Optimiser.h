/**
 * @author Ernesto Casablanca
 * @author Oliver Schön
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Optimiser class.
 */
#pragma once

#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

#include "lucid/lib/eigen.h"

namespace lucid {

/** LP problem to solver to achieve the Fourier barrier synthesis. */
struct FourierBarrierSynthesisProblem {
  static constexpr int num_extra_vars = 10;  ///< Number of extra variables in the Fourier barrier synthesis problem
  static constexpr double tolerance = 1e-8;  ///< Tolerance for strict inequalities

  int num_constraints{1};                    ///< Number of constraints in the LP
  ConstMatrixRefCopy fxn_lattice{Matrix{}};  ///< Lattice obtained from the periodic space
  ConstMatrixRefCopy dn_lattice{
      Matrix{}};  ///< Lattice with the differences between phi(xp) and phi(x) in the periodic space
  const std::vector<Index>& x_include_mask{};   ///< Lattice mask for the points in x
  const std::vector<Index>& x_exclude_mask{};   ///< Lattice mask for the points not in x
  const std::vector<Index>& x0_include_mask{};  ///< Lattice mask for the points in x0
  const std::vector<Index>& x0_exclude_mask{};  ///< Lattice mask for the points not in x0
  const std::vector<Index>& xu_include_mask{};  ///< Lattice mask for the points in xu
  const std::vector<Index>& xu_exclude_mask{};  ///< Lattice mask for the points not in xu
  int T{1};                                     ///< Time horizon
  double gamma{1};                              ///< @gamma_ value
  double eta_coeff{0};                          ///< Coefficient for the eta constraint @f$ 2 / (C - A_x0 + 1) @f$
  double min_x0_coeff{0};    ///< Coefficient for the lower bound on B at x0 @f$ (C - A_{x0} - 1) / (C - A_{x0} + 1) @f$
  double max_sx0_coeff{0};   ///< Coefficient for the difference in B at x0 @f$ A_{x0} / (C - A_{x0} + 1) @f$
  double gamma_coeff{0};     ///< Coefficient for the gamma constraint @f$ 2 / (C - A_{xu} + 1) @f$
  double max_xu_coeff{0};    ///< Coefficient for the upper bound on B at xu @f$ (C - A_{xu} - 1) / (C - A_{xu} + 1) @f$
  double min_sxu_coeff{0};   ///< Coefficient for the difference in B at xu @f$ A_{xu} / (C - A_{xu} + 1) @f$
  double ebk{0};             ///< Coefficient for the Kushner constraint @f$ \epsilon * target\_norm * \kappa @f$
  double c_ebk_coeff{0};     ///< Coefficient for the Kushner constraint @f$ 2 / (C - A_{x} + 1) @f$
  double min_d_coeff{0};     ///< Coefficient for the lower bound on B at x @f$ (C - A_{x} - 1) / (C - A_{x} + 1) @f$
  double max_d_sx_coeff{0};  ///< Coefficient for the difference in B at x @f$ A_{x} / (C - A_{x} + 1) @f$
  double max_x_coeff{0};     ///< Coefficient for the upper bound on B at x @f$ (C - A_{x} - 1) / (C - A_{x} + 1) @f$
  double min_sx_coeff{0};    ///< Coefficient for the difference in B at x @f$ A_{x} / (C - A_{x} + 1) @f$

  std::string to_string() const;
};

/**
 * Base class for optimisation solvers.
 * This class provides a common interface for different optimisation backends
 * (Gurobi, HiGHS, ALGLIB, SoPlex) used in the verification process.
 */
class Optimiser {
 public:
  virtual ~Optimiser() = default;
  /**
   * Callback function called when the optimisation is done.
   * @param success true if the optimisation was successful, false if no solution was found
   * @param obj_val objective value. 0 if no solution was found
   * @param eta eta value. 0 if no solution was found
   * @param c c value. 0 if no solution was found
   * @param norm actual norm of the barrier function. 0 if no solution was found
   */
  using SolutionCallback = std::function<void(bool, double, const Vector&, double, double, double)>;

  /**
   * Construct a new Optimiser object.
   * If `problem_log_file` is provided, the optimiser will log the LP problem to the specified file.
   * If `iis_log_file` is provided, the optimiser will log the
   * Irreducible Inconsistent Subsystem (IIS) to the specified file if the problem is found to be infeasible.
   * @param problem_log_file file to log the problem to
   * @param iis_log_file file to log the IIS to
   */
  explicit Optimiser(std::string problem_log_file = "", std::string iis_log_file = "");

  /**
   * Solve the FourierBarrierCertificate synthesis optimisation problem by solving the corresponding LP.
   * See @ref FourierBarrierCertificate::synthesize for more details.
   * @param problem problem definition
   * @param cb callback function called when the optimisation is done
   * @return true if the optimisation was successful
   * @return false if no solution was found (i.e., the problem is infeasible)
   */
  [[nodiscard]] bool solve_fourier_barrier_synthesis(const FourierBarrierSynthesisProblem& problem,
                                                     const SolutionCallback& cb) const;

  /** @getter{problem log file, solver} */
  [[nodiscard]] const std::string& problem_log_file() const { return problem_log_file_; }
  /** @getter{irreducible infeasible set log file, solver} */
  [[nodiscard]] const std::string& iis_log_file() const { return iis_log_file_; }
  /** @getter{problem log file, solver} */
  [[nodiscard]] bool has_problem_log_file() const { return !problem_log_file_.empty(); }
  /** @getter{irreducible infeasible set log file, solver} */
  [[nodiscard]] bool has_iis_log_file() const { return !iis_log_file_.empty(); }
  /** @getsetter{problem log file, solver} */
  std::string& m_problem_log_file() { return problem_log_file_; }
  /** @getsetter{irreducible infeasible set log file, solver} */
  std::string& m_iis_log_file() { return iis_log_file_; }
  /** @checker{configured to log the lp problem, solver} */
  [[nodiscard]] bool should_log_problem() const { return !problem_log_file_.empty(); }
  /** @checker{configured to log the iis, solver} */
  [[nodiscard]] bool should_log_iis() const { return !iis_log_file_.empty(); }

  /** @to_string */
  [[nodiscard]] virtual std::string to_string() const;

 protected:
  /**
   * Implementation of the Fourier barrier synthesis solver.
   * It interfaces with the specific LP solver backend.
   * See @ref FourierBarrierCertificate::synthesize for more details.
   * @param params problem definition
   * @param cb callback function called when the optimisation is done
   * @return true if the optimisation was successful
   * @return false if no solution was found (i.e., the problem is infeasible
   */
  [[nodiscard]] virtual bool solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem& params,
                                                                  const SolutionCallback& cb) const = 0;

  std::string problem_log_file_;  ///< File to log the problem to, if provided
  std::string iis_log_file_;      ///< File to log the IIS (Irreducible Inconsistent Subsystem) to, if found
};

std::ostream& operator<<(std::ostream& os, const FourierBarrierSynthesisProblem& problem);
std::ostream& operator<<(std::ostream& os, const Optimiser& optimiser);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Optimiser)

#endif
