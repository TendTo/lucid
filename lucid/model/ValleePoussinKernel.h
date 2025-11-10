/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * VallePoussinKernel class.
 */
#pragma once

#include <iosfwd>

#include "lucid/model/Kernel.h"

namespace lucid {

/**
 * Vallée-Poussin kernel.
 * Given a vector space @XsubRn and a vector @f$ x \in \mathcal{X} @f$, the Vallee-Poussin kernel
 * @f$ k_{a,b}^n : \mathcal{X} \to \mathbb{R} @f$ with parameters @f$ a, b \in \mathbb{R} @f$
 * is defined as
 * @f[
 * k_{a,b}^n(x) = \frac{1}{(b - a)^n} \prod_{i=1}^{n} \frac{
 * \sin\left( \frac{b + a}{2} x_i \right)
 * \sin\left( \frac{b - a}{2} x_i \right)
 * }{
 * \sin^2{\left( \frac{x_i}{2} \right)}
 * }
 * @f]
 */
class ValleePoussinKernel final : public Kernel {
 public:
  using Kernel::set;
  /**
   * Construct a new ValleePoussinKernel object with the given parameters.
   * @param a parameter
   * @param b parameter
   */
  explicit ValleePoussinKernel(double a = 1.0, double b = 1.0);

  [[nodiscard]] bool is_stationary() const override { return true; }

  /** @getter{a parameter, kernel} */
  [[nodiscard]] double a() const { return a_; }
  /** @getter{b parameter, kernel} */
  [[nodiscard]] double b() const { return b_; }

  [[nodiscard]] double get_d(Parameter parameter) const override;
  void set(Parameter parameter, double value) override;

  [[nodiscard]] std::unique_ptr<Kernel> clone() const override;

 private:
  Matrix apply_impl(ConstMatrixRef x1, ConstMatrixRef x2, std::vector<Matrix>* gradient) const override;

  double a_;  ///< 'a' parameter
  double b_;  ///< 'b' parameter
};

std::ostream& operator<<(std::ostream& os, const ValleePoussinKernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::ValleePoussinKernel)

#endif
