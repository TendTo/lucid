/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GurobiLinearOptimiser class.
 */
#pragma once

#include "lucid/lib/eigen.h"

namespace lucid {

/**
 * Linear optimiser using the Gurobi solver.
 */
class GurobiLinearOptimiser {
 public:
  /**
   * Callback function called when the optimisation is done.
   * @param success true if the optimisation was successful, false if no solution was found
   * @param obj_val objective value. 0 if no solution was found
   * @param eta eta value. 0 if no solution was found
   * @param c c value. 0 if no solution was found
   * @param norm actual norm of the barrier function. 0 if no solution was found
   */
  using SolutionCallback = std::function<void(bool, double, double, double, double)>;

  /**
   * Construct a new GurobiLinearOptimiser object.
   * @param T time horizon
   * @param gamma gamma value
   * @param epsilon epsilon value
   * @param b_norm norm of the barrier function
   * @param b_kappa kappa value
   * @param sigma_f sigma_f value
   */
  GurobiLinearOptimiser(const int T, const double gamma, const double epsilon, const double b_norm,
                        const double b_kappa, const double sigma_f)
      : T_{T}, gamma_{gamma}, epsilon_{epsilon}, b_norm_{b_norm}, b_kappa_{b_kappa}, sigma_f_{sigma_f} {}

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
   * @param callback callback function
   * @return true if the optimisation was successful
   * @return false if no solution was found
   */
  bool solve(ConstMatrixRef f0_lattice, ConstMatrixRef fu_lattice, ConstMatrixRef phi_mat, ConstMatrixRef w_mat,
             Dimension rkhs_dim, Dimension num_frequencies_per_dim, Dimension num_frequency_samples_per_dim,
             Dimension original_dim, const SolutionCallback& callback) const;

 private:
  const int T_;           ///< Time horizon
  const double gamma_;    ///< Gamma value
  const double epsilon_;  ///< Epsilon value
  const double b_norm_;   ///< Norm of the barrier function
  const double b_kappa_;  ///< Kappa value
  const double sigma_f_;  ///< Sigma_f value
};

}  // namespace lucid
