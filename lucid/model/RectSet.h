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
  using Set::change_size;
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
  [[nodiscard]] Vector general_lower_bound() const override { return lb_; }
  /** @getter{upper bound, rectangular set} */
  [[nodiscard]] const Vector& upper_bound() const { return ub_; }
  [[nodiscard]] Vector general_upper_bound() const override { return ub_; }
  /** @getter{size for each dimension, rectangular set} */
  [[nodiscard]] Vector sizes() const { return ub_ - lb_; }

  [[nodiscard]] Matrix sample(Index num_samples) const override;

  [[nodiscard]] bool operator()(ConstVectorRef x) const override;

  /**
   * Generate a lattice of points in the set.
   * In python, this would be implemented as:
   * @code{.py}
   * import numpy as np
   *
   * def build_lattice(points_per_dim, endpoint):
   *    # self.lb_ and self.ub_ are the lower and upper bounds of the rectangular set
   *    grids = [np.linspace(l, u, n, endpoint) for l, u, n in zip(self.lb_, self.ub_, points_per_dim)]
   *    mesh = np.meshgrid(*grids, indexing="xy")
   *    pts = np.vstack([m.ravel() for m in mesh]).T
   *    return pts
   * @endcode
   * @param points_per_dim number of points per each dimension
   * @param endpoint whether to include the endpoints of the lattice
   * @return lattice of points in the set
   */
  [[nodiscard]] Matrix lattice(const VectorI& points_per_dim, bool endpoint) const override;

  void change_size(ConstVectorRef delta_size) override;

  [[nodiscard]] std::unique_ptr<Set> to_rect_set() const override;

  /**
   * Compute the rectangular set relative to another rectangular set.
   * Instead of absolute coordinates, the new rectangular set will be expressed
   * in coordinates relative to the lower bound of the other rectangular set.
   * @pre The two rectangular sets must have the same dimension.
   * @param set other rectangular set
   * @return new rectangular set expressed in relative coordinates
   */
  [[nodiscard]] RectSet relative_to(const RectSet& set) const;
  /**
   * Compute the rectangular set relative a new origin point.
   * Instead of absolute coordinates, the new rectangular set will be expressed
   * in coordinates relative to the given point.
   * @pre The point must have the same dimension as the rectangular set.
   * @param point point representing the new origin
   * @return new rectangular set expressed in relative coordinates
   */
  [[nodiscard]] RectSet relative_to(ConstVectorRef point) const;

  /**
   * Scale the rectangular set by the given factor(s).
   * The scaling is performed with respect to the center of the rectangular set.
   * @param scale scaling factor(s) for each dimension
   * @return new scaled rectangular set
   */
  [[nodiscard]] RectSet scale(ConstVectorRef scale) const;
  /**
   * Scale the rectangular set by the given factor.
   * The scaling is performed with respect to the center of the rectangular set.
   * @param scale scaling factor
   * @return new scaled rectangular set
   */
  [[nodiscard]] RectSet scale(double scale) const;
  /**
   * Scale the rectangular set by the given factor(s) while keeping it inside the given bounds.
   * The scaling is performed with respect to the center of the rectangular set.
   * The scaling factor can be computed relative to either
   * - the current size of the rectangular set;
   * - the size of the bounding rectangular set.
   * @param scale scaling factor(s) for each dimension
   * @param bounds bounding rectangular set
   * @param relative_to_bounds if true, the scaling factor is computed relative to the size of the bounding
   * rectangular set; if false, the scaling factor is computed relative to the current size of the rectangular
   * @return new scaled rectangular set
   */
  [[nodiscard]] RectSet scale(ConstVectorRef scale, const RectSet& bounds, bool relative_to_bounds = false) const;
  /**
   * Scale the rectangular set by the given factor while keeping it inside the given bounds.
   * The scaling is performed with respect to the center of the rectangular set.
   * The scaling factor can be computed relative to either
   * - the current size of the rectangular set;
   * - the size of the bounding rectangular set.
   * @param scale scaling factor
   * @param bounds bounding rectangular set
   * @param relative_to_bounds if true, the scaling factor is computed relative to the size of the bounding
   * rectangular set; if false, the scaling factor is computed relative to the current size of the rectangular
   * @return new scaled rectangular set
   */
  [[nodiscard]] RectSet scale(double scale, const RectSet& bounds, bool relative_to_bounds = false) const;

  RectSet& operator+=(ConstVectorRef offset);
  RectSet& operator+=(Scalar offset);
  [[nodiscard]] RectSet operator+(ConstVectorRef offset) const;
  [[nodiscard]] RectSet operator+(Scalar offset) const;
  RectSet& operator-=(ConstVectorRef offset);
  RectSet& operator-=(Scalar offset);
  [[nodiscard]] RectSet operator-(ConstVectorRef offset) const;
  [[nodiscard]] RectSet operator-(Scalar offset) const;
  RectSet& operator*=(ConstVectorRef scale);
  RectSet& operator*=(Scalar scale);
  [[nodiscard]] RectSet operator*(ConstVectorRef scale) const;
  [[nodiscard]] RectSet operator*(Scalar scale) const;
  RectSet& operator/=(ConstVectorRef scale);
  RectSet& operator/=(Scalar scale);
  [[nodiscard]] RectSet operator/(Scalar scale) const;
  [[nodiscard]] RectSet operator/(ConstVectorRef scale) const;
  bool operator==(const Set& other) const override;
  bool operator==(const RectSet& other) const;

  /**
   * Convert the rectangular set to a matrix representation.
   * @return matrix where the first row contains lower bounds and the second row contains upper bounds
   */
  operator Matrix() const;

  [[nodiscard]] std::string to_string() const override;

 private:
  [[nodiscard]] std::unique_ptr<Set> scale_wrapped_impl(ConstVectorRef scale, const RectSet& bounds,
                                                        bool relative_to_bounds) const override;

  Vector lb_;  ///< Lower bound vector
  Vector ub_;  ///< Upper bound vector
};

std::ostream& operator<<(std::ostream& os, const RectSet& set);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::RectSet)

#endif
