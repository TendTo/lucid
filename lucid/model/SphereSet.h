/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * SphereSet class.
 */
#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "lucid/model/Set.h"

namespace lucid {

/**
 * Multidimensional sphere set.
 * A vector @x is in the set if @f$ ||x - c||_2 \le r @f$,
 * where @f$ c @f$ is the center and @f$ r @f$ is the radius.
 * The sampling is uniform over the volume of the sphere.
 * The samples are generated using the
 * [Muller method](https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/).
 */
class SphereSet final : public Set {
 public:
  /**
   * Construct a sphere set from a `center` and a `radius`.
   * The dimension of the space the sphere set lives in is determined by the size of the `center` vector.
   * @param center vector representing the center of the sphere
   * @param radius radius of the sphere
   */
  SphereSet(ConstVectorRef center, Scalar radius);
  [[nodiscard]] Dimension dimension() const override { return center_.size(); }
  [[nodiscard]] Matrix sample(Index num_samples) const override;
  [[nodiscard]] bool operator()(ConstVectorRef x) const override;
  [[nodiscard]] Matrix lattice(const VectorI &points_per_dim, bool endpoint) const override;

  /** @getter{center, sphere set} */
  [[nodiscard]] const Vector &center() const { return center_; }
  /** @getter{radius, sphere set} */
  [[nodiscard]] Scalar radius() const { return radius_; }

  void change_size(ConstVectorRef delta_size) override;

  [[nodiscard]] Vector general_lower_bound() const override;
  [[nodiscard]] Vector general_upper_bound() const override;

  [[nodiscard]] std::unique_ptr<Set> to_rect_set() const override;

  [[nodiscard]] std::string to_string() const override;

  bool operator==(const Set &other) const override;
  bool operator==(const SphereSet &other) const;

 private:
  [[nodiscard]] std::unique_ptr<Set> increase_size_impl(ConstVectorRef size_increase) const override;

  Vector center_;  ///< Center of the sphere. Determines the dimension of the sphere set
  Scalar radius_;  ///< Radius of the sphere
};

std::ostream &operator<<(std::ostream &os, const SphereSet &set);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::SphereSet)

#endif
