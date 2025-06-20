/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GurobiLinearOptimiser class.
 */
#pragma once

#include <string>

#include "lucid/lib/eigen.h"
#include "lucid/verification/Optimiser.h"

namespace lucid {

/**
 * Linear optimiser using the Gurobi solver.
 */
class GurobiLinearOptimiser final : public Optimiser {
 public:
  /**
   * Construct a new GurobiLinearOptimiser object.
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
  GurobiLinearOptimiser(int T, double gamma, double epsilon, double b_norm, double b_kappa, double sigma_f,
                        double C_coeff = 1.0, std::string problem_log_file = "", std::string iis_log_file = "");

  /**
   * Solve the linear optimisation
   * @param f0_lattice lattice obtained from the initial set after applying the feature map
   * @param fu_lattice lattice obtained from the unsafe set after applying the feature map
   * @param phi_mat phi matrix
   * @param w_mat weight matrix
   * @param rkhs_dim dimension of the RKHS
   * @param num_frequencies_per_dim number of frequencies per dimension
   * @param num_frequency_samples_per_dim number of frequency samples per dimension
   * @param original_dim original dimension
   * @param cb callback function
   * @return true if the optimisation was successful
   * @return false if no solution was found
   */
  [[nodiscard]] bool solve(ConstMatrixRef f0_lattice, ConstMatrixRef fu_lattice, ConstMatrixRef phi_mat,
                           ConstMatrixRef w_mat, Dimension rkhs_dim, Dimension num_frequencies_per_dim,
                           Dimension num_frequency_samples_per_dim, Dimension original_dim,
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

 private:
  const int T_;                   ///< Time horizon
  const double gamma_;            ///< Gamma value
  const double epsilon_;          ///< Epsilon value
  const double b_norm_;           ///< Norm of the barrier function
  const double b_kappa_;          ///< Kappa value
  const double sigma_f_;          ///< Sigma_f value
  const double C_coeff_;          ///< Coefficient to apply to `C` for the barrier function
  std::string problem_log_file_;  ///< File to log the problem to
  std::string iis_log_file_;      ///< File to log the IIS (Irreducible Inconsistent Subsystem) to, if found
};

}  // namespace lucid
