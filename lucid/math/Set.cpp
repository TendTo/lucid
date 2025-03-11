/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/Set.h"

#include <ostream>

namespace lucid {

Vector Set::sample_element() const {
  Matrix samples = sample_element(1);
  return samples.row(0);
}
Matrix Set::lattice(const Index points_per_dim, const bool include_endpoints) const {
  return lattice(Eigen::VectorX<Index>::Constant(dimension(), points_per_dim), include_endpoints);
}

std::ostream& operator<<(std::ostream& os, const Set&) { return os << "Set"; }

}  // namespace lucid
