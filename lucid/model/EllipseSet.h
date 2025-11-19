/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * EllipseSet class.
 */
#pragma once

#include <iosfwd>

#include "lucid/model/Set.h"

namespace lucid {

/**
 * Multidimensional ellipsoid set.
 * A vector @x is in the set if @f$ \sum_{i=1}^{d} \left(\frac{x_i - c_i}{r_i}\right)^2 \le 1 @f$,
 * where @f$ c @f$ is the center and @f$ r @f$ is the vector of semi-axes (radii).
 */
class EllipseSet final : public Set {
 public:
  /**
   * Construct an ellipsoid set from a `center` and a vector of `radii`.
   * The dimension of the space the ellipsoid set lives in is determined by the size of the `center` vector.
   * @param center vector representing the center of the ellipsoid
   * @param radii vector of semi-axes (radii) for each dimension
   */
  EllipseSet(ConstVectorRef center, ConstVectorRef radii);

  /**
   * Construct an ellipsoid set from a `center` and a uniform `radius`.
   * This creates a sphere (all radii are equal).
   * The dimension of the space the ellipsoid set lives in is determined by the size of the `center` vector.
   * @param center vector representing the center of the ellipsoid
   * @param radius uniform radius for all dimensions
   */
  EllipseSet(ConstVectorRef center, Scalar radius);

  [[nodiscard]] Dimension dimension() const override { return center_.size(); }
  [[nodiscard]] Matrix lattice(const VectorI& points_per_dim, bool endpoint) const override;
  [[nodiscard]] Matrix sample(Index num_samples) const override;
  [[nodiscard]] bool operator==(const EllipseSet& other) const;
  [[nodiscard]] bool operator==(const Set& other) const override;
  [[nodiscard]] bool operator()(ConstVectorRef x) const override;

  /** @getter{center, ellipsoid set} */
  [[nodiscard]] const Vector& center() const { return center_; }
  /** @getter{radii, ellipsoid set} */
  [[nodiscard]] const Vector& radii() const { return radii_; }

  [[nodiscard]] Vector general_lower_bound() const override;
  [[nodiscard]] Vector general_upper_bound() const override;

  void change_size(ConstVectorRef delta_size) override;

  [[nodiscard]] std::unique_ptr<Set> to_rect_set() const override;

  [[nodiscard]] std::string to_string() const override;

 private:
  Vector center_;  ///< Center of the ellipsoid. Determines the dimension of the ellipsoid set
  Vector radii_;   ///< Semi-axes (radii) for each dimension
};

std::ostream& operator<<(std::ostream& os, const EllipseSet& set);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::EllipseSet)

#endif
