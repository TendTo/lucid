/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * RectSet class.
 */
#pragma once

#include <string>
#include <vector>

#include "lucid/model/Set.h"

namespace lucid {

/**
 * Rectangular set over an arbitrary number of dimensions.
 * A vector @x is in the set if @f$ lb_i \le x_i \le ub_i @f$ for all components of the vector.
 */
class RectSet final : public Set {
 public:
  using Set::lattice;
  /**
   * Construct a rectangular set from lower and upper bounds.
   * Both bounds must belong to the same vector space.
   * @param lb lower bound vector
   * @param ub upper bound vector
   */
  RectSet(Vector lb, Vector ub);
  /**
   * Construct a rectangular set from lower and upper bounds.
   * Both bounds must belong to the same vector space.
   * @param lb lower bound vector
   * @param ub upper bound vector
   */
  RectSet(std::initializer_list<Scalar> lb, std::initializer_list<Scalar> ub);
  /**
   * Construct a rectangular set from lower and upper bounds.
   * Both bounds must belong to the same vector space.
   * @param bounds vector of pairs of lower and upper bounds
   */
  explicit RectSet(const std::vector<std::pair<Scalar, Scalar>>& bounds);

  /**
   * Construct a rectangular set from lower and upper bounds.
   * Both bounds must belong to the same vector space.
   * @param bounds vector of pairs of lower and upper bounds
   */
  RectSet(std::initializer_list<std::pair<Scalar, Scalar>> bounds);

  [[nodiscard]] Dimension dimension() const override { return lb_.size(); }
  /** @getter{lower bound, rectangular set} */
  [[nodiscard]] const Vector& lower_bound() const { return lb_; }
  /** @getter{upper bound, rectangular set} */
  [[nodiscard]] const Vector& upper_bound() const { return ub_; }

  [[nodiscard]] Matrix sample(Index num_samples) const override;

  [[nodiscard]] bool operator()(ConstVectorRef x) const override;

  [[nodiscard]] Matrix lattice(const VectorI& points_per_dim, bool include_endpoints) const override;

  /**
   * Convert the rectangular set to a matrix representation.
   * @return matrix where the first row contains lower bounds and the second row contains upper bounds
   */
  operator Matrix() const;

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
