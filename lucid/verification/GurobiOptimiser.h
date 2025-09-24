/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GurobiOptimiser class.
 */
#pragma once

#include <utility>

#include "lucid/lib/eigen.h"
#include "lucid/verification/Optimiser.h"

namespace lucid {

/**
 * Linear optimiser using the [Gurobi](https://www.gurobi.com/) solver.
 */
class GurobiOptimiser final : public Optimiser {
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
  bool solve(ConstMatrixRef f0_lattice, ConstMatrixRef fu_lattice, ConstMatrixRef phi_mat, ConstMatrixRef w_mat,
             Dimension rkhs_dim, Dimension num_frequencies_per_dim, Dimension num_frequency_samples_per_dim,
             Dimension original_dim, const SolutionCallback& cb) const;

  /**
   * Compute the bounding box of a polytope defined by Ax <= b.
   * Uses linear programming to find the minimum and maximum values for each dimension.
   * @param A constraint matrix where each row represents a halfspace
   * @param b constraint vector of right-hand side values
   * @return pair of vectors (lower_bounds, upper_bounds) for each dimension
   */
  static std::pair<Vector, Vector> bounding_box(ConstMatrixRef A, ConstVectorRef b);

 private:
  bool solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisParameters& params,
                                            const SolutionCallback& cb) const override;
};

}  // namespace lucid
