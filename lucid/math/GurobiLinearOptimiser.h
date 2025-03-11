/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GurobiLinearOptimiser class.
 */
#pragma once

#include "lucid/lib/eigen.h"

namespace lucid {

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

  GurobiLinearOptimiser(int T, double gamma, double epsilon, double b_norm, double b_kappa, double sigma_f)
      : T_{T}, gamma_{gamma}, epsilon_{epsilon}, b_norm_{b_norm}, b_kappa_{b_kappa}, sigma_f_{sigma_f} {}

  bool solve(ConstMatrixRef f0_lattice, ConstMatrixRef fu_lattice, ConstMatrixRef phi_mat, ConstMatrixRef w_mat,
             Dimension rkhs_dim, Dimension num_frequencies_per_dim, Dimension num_frequency_samples_per_dim,
             Dimension original_dim, const SolutionCallback& callback);

 private:
  const int T_;
  const double gamma_;
  const double epsilon_;
  const double b_norm_;
  const double b_kappa_;
  const double sigma_f_;
};

}  // namespace lucid
