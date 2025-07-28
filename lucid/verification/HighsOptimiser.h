/**
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * HighsOptimiser class.
 */
#pragma once

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
};

}  // namespace lucid
