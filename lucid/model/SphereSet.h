/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * SphereSet class.
 */
#pragma once

#include <iosfwd>

#include "lucid/model/Set.h"

namespace lucid {

class SphereSet final : public Set {
 public:
  SphereSet(ConstVectorRef center, Scalar radius);
  [[nodiscard]] Dimension dimension() const override { return center_.size(); }
  [[nodiscard]] Matrix sample(Index num_samples) const override;
  [[nodiscard]] bool operator()(ConstVectorRef x) const override;
  [[nodiscard]] Matrix lattice(const VectorI &points_per_dim, bool include_endpoints) const override;

  /** @getter{center, sphere set} */
  [[nodiscard]] const Vector &center() const { return center_; }
  /** @getter{radius, sphere set} */
  [[nodiscard]] Scalar radius() const { return radius_; }

 private:
  Vector center_;
  Scalar radius_;
};

std::ostream &operator<<(std::ostream &os, const SphereSet &set);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::SphereSet)

#endif
