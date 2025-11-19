/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * AlglibOptimiser class.
 */
#pragma once

#include <iosfwd>
#include <utility>

#include "lucid/lib/eigen.h"
#include "lucid/verification/Optimiser.h"

namespace lucid {

/**
 * Linear optimiser using the [Alglib](https://www.alglib.net/) mathematical library.
 */
class AlglibOptimiser final : public Optimiser {
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

  static std::pair<Vector, Vector> bounding_box(ConstMatrixRef A, ConstVectorRef b);

 private:
  bool solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem& params,
                                            const SolutionCallback& cb) const override;
};

std::ostream& operator<<(std::ostream& os, const AlglibOptimiser& optimiser);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::AlglibOptimiser)

#endif
