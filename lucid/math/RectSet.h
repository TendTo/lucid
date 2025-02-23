/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * RectSet class.
 */
#pragma once

#include <string>

#include "lucid/math/Set.h"

namespace lucid {

/**
 * Rectangular set over an arbitrary number of dimensions.
 * A vector @x is in the set if @f$ lb_i \le x_i \le ub_i @f$ for all components of the vector.
 */
class RectSet final : public Set {
 public:
  /**
   * Construct a rectangular set from lower and upper bounds.
   * Both bounds must belong to the same vector space.
   * @param lb lower bound vector
   * @param ub upper bound vector
   * @param seed random seed used of the sampling from this moment forward. If negative, the seed is not set
   */
  RectSet(Vector lb, Vector ub, int seed = -1);

  [[nodiscard]] Dimension dimension() const override { return lb_.size(); }
  /** @getter{lower bound, rectangular set} */
  [[nodiscard]] const Vector& lower_bound() const { return lb_; }
  /** @getter{upper bound, rectangular set} */
  [[nodiscard]] const Vector& upper_bound() const { return ub_; }

  [[nodiscard]] Matrix sample_element(int num_samples) const override;

  [[nodiscard]] bool operator()(ConstMatrixRef x) const override;

  void plot(const std::string& color) const override;

 private:
  Vector lb_;  ///< Lower bound vector
  Vector ub_;  ///< Upper bound vector
};

std::ostream& operator<<(std::ostream& os, const RectSet& set);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::RectSet)

#endif
