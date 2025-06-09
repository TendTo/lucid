/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Set.h"

#include <ostream>

#include "lucid/model/MultiSet.h"
#include "lucid/model/RectSet.h"
#include "lucid/util/error.h"

namespace lucid {

Vector Set::sample() const {
  return sample(1l).row(0)
}
Matrix Set::lattice(const Index points_per_dim, const bool include_endpoints) const {
  return lattice(VectorI::Constant(dimension(), points_per_dim), include_endpoints);
}

std::ostream& operator<<(std::ostream& os, const Set& set) {
  if (const auto* casted_set = dynamic_cast<const RectSet*>(&set)) return os << *casted_set;
  if (const auto* casted_set = dynamic_cast<const MultiSet*>(&set)) return os << *casted_set;
  return os << "Set( )";
}

}  // namespace lucid
