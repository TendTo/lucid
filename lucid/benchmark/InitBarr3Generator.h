/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Barrier3Generator class.
 */
#pragma once

#include "lucid/benchmark/Generator.h"
#include "lucid/math/MultiSet.h"
#include "lucid/math/RectSet.h"

namespace lucid::benchmark {

class InitBarr3Generator final : public Generator {
 public:
  [[nodiscard]] constexpr Dimension dimension() const override { return 2; }
  [[nodiscard]] Matrix operator()(ConstMatrixRef x) const override;
  [[nodiscard]] constexpr int num_steps() const override { return 10; }
  [[nodiscard]] constexpr double desired_confidence() override { return 0.9; }
  [[nodiscard]] const Set& initial_set() const override { return initial_set_; }
  [[nodiscard]] const Set& unsafe_set() const override { return unsafe_set_; }
  [[nodiscard]] const Set& set() const override { return set_; }

 private:
  static Matrix f_stoch(ConstMatrixRef x);
  static Matrix f_det(ConstMatrixRef x);

  MultiSet initial_set_{RectSet{Vector2{1, -0.5}, Vector2{2, 0.5}},        //
                        RectSet{Vector2{-1.8, -0.1}, Vector2{-1.2, 0.1}},  //
                        RectSet{Vector2{-1.4, -0.5}, Vector2{-1.2, 0.1}}};
  MultiSet unsafe_set_{RectSet{Vector2{0.4, 0.1}, Vector2{0.6, 0.5}},  //
                       RectSet{Vector2{0.4, 0.1}, Vector2{0.8, 0.3}}};
  RectSet set_{Vector2{-3, -2}, Vector2{2.5, 1}};
};

}  // namespace lucid::benchmark
