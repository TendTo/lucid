/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Barrier3Scenario class.
 */
#pragma once

#include "lucid/benchmark/Scenario.h"
#include "lucid/math/MultiSet.h"
#include "lucid/math/RectSet.h"

namespace lucid::benchmark {

/**
 * Generate a problem in a 2D space for the solver.
 */
class InitBarr3Scenario final : public Scenario {
 public:
  [[nodiscard]] constexpr int num_supp_per_dim() const { return 12; }
  [[nodiscard]] constexpr int num_freq_per_dim() const { return 6; }
  [[nodiscard]] constexpr double sigma_f() const { return 19.456; }
  [[nodiscard]] constexpr double b_norm() const { return 25; }
  [[nodiscard]] constexpr double kappa_b() const { return 1.0; }
  [[nodiscard]] constexpr double gamma() const { return 18.312; }
  [[nodiscard]] const Vector& sigma_l() const;
  [[nodiscard]] constexpr int T() const { return 10; }
  [[nodiscard]] constexpr double lambda() const { return 1e-5; }
  [[nodiscard]] constexpr int N() const { return 1000; }
  [[nodiscard]] constexpr double epsilon() const { return 0.0033; }
  [[nodiscard]] constexpr bool autonomous() const { return true; }

  [[nodiscard]] const Matrix& x_samples() const;
  [[nodiscard]] const Matrix& xp_samples() const;
  [[nodiscard]] const Matrix& x_limits() const;

  [[nodiscard]] constexpr Dimension dimension() const override { return 2; }
  [[nodiscard]] Matrix operator()(ConstMatrixRef x) const override;
  [[nodiscard]] constexpr int num_steps() const override { return 10; }
  [[nodiscard]] constexpr double desired_confidence() override { return 0.9; }
  [[nodiscard]] const Set& initial_set() const override { return initial_set_; }
  [[nodiscard]] const Set& unsafe_set() const override { return unsafe_set_; }
  [[nodiscard]] const Set& set() const override { return set_; }

 private:
  /**
   * Stochastic transition function.
   * It wraps the deterministic transition function @ref f_det adding a small noise to the result.
   * @param x @nvector from @X
   * @return @nvector @f$ f(x) + \text{noise} @f$
   */
  static Matrix f_stoch(ConstMatrixRef x);
  /**
   * Deterministic transition function.
   * It applies the following transformation:
   * @f[
   * f(\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}) = \begin{bmatrix} x_2 \\ -x_1 - x_2 + \frac{1}{3} x_1^3 \end{bmatrix} .
   * @f]
   * @param x @nvector from @X
   * @return @nvector @fx
   */
  static Matrix f_det(ConstMatrixRef x);

  MultiSet initial_set_{RectSet{Vector2{1, -0.5}, Vector2{2, 0.5}},         //
                        RectSet{Vector2{-1.8, -0.1}, Vector2{-1.2, 0.1}},   //
                        RectSet{Vector2{-1.4, -0.5}, Vector2{-1.2, 0.1}}};  ///< Initial set.
  MultiSet unsafe_set_{RectSet{Vector2{0.4, 0.1}, Vector2{0.6, 0.5}},       //
                       RectSet{Vector2{0.4, 0.1}, Vector2{0.8, 0.3}}};      ///< Unsafe set.
  RectSet set_{Vector2{-3, -2}, Vector2{2.5, 1}};                           ///< Set of interest.
};

}  // namespace lucid::benchmark
