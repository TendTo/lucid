/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/Set.h"

#include <ostream>

#include "lucid/math/MultiSet.h"
#include "lucid/math/RectSet.h"
#include "lucid/util/error.h"

namespace lucid {

Vector Set::sample() const {
  Matrix samples = sample(1l);
  return samples.row(0);
}
Matrix Set::lattice(const Index points_per_dim, const bool include_endpoints) const {
  return lattice(Eigen::VectorX<Index>::Constant(dimension(), points_per_dim), include_endpoints);
}

std::ostream& operator<<(std::ostream& os, const Set& set) {
  if (dynamic_cast<const RectSet*>(&set)) {
    return os << static_cast<const RectSet&>(set);
  }
  if (dynamic_cast<const MultiSet*>(&set)) {
    return os << static_cast<const MultiSet&>(set);
  }
  LUCID_UNREACHABLE();
}

}  // namespace lucid
