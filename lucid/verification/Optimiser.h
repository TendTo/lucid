/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Optimiser class.
 */
#pragma once

#include <functional>
#include <string>

#include "lucid/lib/eigen.h"

namespace lucid {

/** LP problem to solver to achieve the Fourier barrier synthesis. */
struct FourierBarrierSynthesisProblem {
  Dimension num_vars;              ///< Number of variables in the LP
  Dimension num_constraints;       ///< Number of constraints in the LP
  ConstMatrixRefCopy fxn_lattice;  ///< Lattice obtained from the periodic space
  ConstMatrixRefCopy dn_lattice;   ///< Lattice with the differences between phi(xp) and phi(x) in the periodic space
  const std::vector<Index>& x_include_mask;   ///< Lattice mask for the points in x
  const std::vector<Index>& x_exclude_mask;   ///< Lattice mask for the points not in x
  const std::vector<Index>& x0_include_mask;  ///< Lattice mask for the points in x0
  const std::vector<Index>& x0_exclude_mask;  ///< Lattice mask for the points not in x0
  const std::vector<Index>& xu_include_mask;  ///< Lattice mask for the points in xu
  const std::vector<Index>& xu_exclude_mask;  ///< Lattice mask for the points not in xu
  int T;                                      ///< Time horizon
  double gamma;                               ///< @gamma value
  double C;                                   ///< @f$ C = (1 - \frac{2 f_\max}{\tilde{Q}}^{\frac{-n}{2}} @f$
  double b_kappa;
  double eta_coeff;        ///< Coefficient for the eta constraint @f$ 2 / (C - A_x0 + 1) @f$
  double min_x0_coeff;     ///< Coefficient for the lower bound on B at x0 @f$ (C - A_{x0} - 1) / (C - A_{x0} + 1) @f$
  double diff_sx0_coeff;   ///< Coefficient for the difference in B at x0 @f$ A_{x0} / (C - A_{x0} + 1) @f$
  double gamma_coeff;      ///< Coefficient for the gamma constraint @f$ 2 / (C - A_{xu} + 1) @f$
  double max_xu_coeff;     ///< Coefficient for the upper bound on B at xu @f$ (C - A_{xu} - 1) / (C - A_{xu} + 1) @f$
  double diff_sxu_coeff;   ///< Coefficient for the difference in B at xu @f$ A_{xu} / (C - A_{xu} + 1) @f$
  double ebk;              ///< Coefficient for the Kushner constraint @f$ \epsilon * target\_norm * \kappa @f$
  double c_ebk_coeff;      ///< Coefficient for the Kushner constraint @f$ 2 / (C - A_{x} + 1) @f$
  double min_d_coeff;      ///< Coefficient for the lower bound on B at x @f$ (C - A_{x} - 1) / (C - A_{x} + 1) @f$
  double diff_d_sx_coeff;  ///< Coefficient for the difference in B at x @f$ A_{x} / (C - A_{x} + 1) @f$
  double max_x_coeff;      ///< Coefficient for the upper bound on B at x @f$ (C - A_{x} - 1) / (C - A_{x} + 1) @f$
  double diff_sx_coeff;    ///< Coefficient for the difference in B at x @f$ A_{x} / (C - A_{x} + 1) @f$
  double fctr1;
  double fctr2;
  double unsafe_rhs;
  double kushner_rhs;
  double A_x;
  double A_x0;
  double A_xu;
};

/**
 * Base class for optimisation solvers.
 * This class provides a common interface for different optimisation backends
 * (Gurobi, HiGHS, ALGLIB) used in the verification process.
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
   * @pre `T` must be greater than 0
   * @pre If provided, `problem_log_file` and `iis_log_file` must be valid file paths.
   * The former must have the extension `.lp` or `.mps`, and the latter must have the extension `.ilp`.
   * @param T time horizon
   * @param gamma gamma value
   * @param epsilon epsilon value
   * @param b_norm norm of the barrier function
   * @param b_kappa kappa value
   * @param sigma_f sigma_f value
   * @param problem_log_file file to log the problem to. If empty, no logging is done
   * @param iis_log_file file to log the irreducible infeasible set (IIS) to, if found. If empty, no logging is done
   */
  Optimiser(int T, double gamma, double epsilon, double b_norm, double b_kappa, double sigma_f, double C_coeff = 1.0,
            std::string problem_log_file = "", std::string iis_log_file = "");
  explicit Optimiser(std::string problem_log_file = "", std::string iis_log_file = "");

  bool solve_fourier_barrier_synthesis(const FourierBarrierSynthesisProblem& params, const SolutionCallback& cb) const;

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
  /** @getter{time horizon, solver} */
  [[nodiscard]] int T() const { return T_; }
  /** @getter{gamma, solver} */
  [[nodiscard]] double gamma() const { return gamma_; }
  /** @getter{epsilon, solver} */
  [[nodiscard]] double epsilon() const { return epsilon_; }
  /** @getter{b_norm, solver} */
  [[nodiscard]] double b_norm() const { return b_norm_; }
  /** @getter{kappa, solver} */
  [[nodiscard]] double b_kappa() const { return b_kappa_; }
  /** @getter{sigma_f, solver} */
  [[nodiscard]] double sigma_f() const { return sigma_f_; }
  /** @getter{C coefficient, solver} */
  [[nodiscard]] double C_coeff() const { return C_coeff_; }
  /** @checker{configured to log the lp problem, solver} */
  [[nodiscard]] bool should_log_problem() const { return !problem_log_file_.empty(); }
  /** @checker{configured to log the iis, solver} */
  [[nodiscard]] bool should_log_iis() const { return !iis_log_file_.empty(); }

 protected:
  virtual bool solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem& params,
                                                    const SolutionCallback& cb) const = 0;

  const int T_;                   ///< Time horizon
  const double gamma_;            ///< Gamma value
  const double epsilon_;          ///< Epsilon value
  const double b_norm_;           ///< Norm of the barrier function
  const double b_kappa_;          ///< Kappa value
  const double sigma_f_;          ///< Sigma_f value
  const double C_coeff_;          ///< Coefficient to apply to `C` for the barrier function
  std::string problem_log_file_;  ///< File to log the problem to, if provided
  std::string iis_log_file_;      ///< File to log the IIS (Irreducible Inconsistent Subsystem) to, if found
};

}  // namespace lucid
