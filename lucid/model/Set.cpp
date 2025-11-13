/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Set.h"

#include <ostream>

#include "lucid/model/MultiSet.h"
#include "lucid/model/PolytopeSet.h"
#include "lucid/model/RectSet.h"
#include "lucid/model/SphereSet.h"
#include "lucid/util/error.h"

namespace lucid {

Vector Set::sample() const { return sample(1l).row(0); }

Matrix Set::include(ConstMatrixRef xs) const { return xs(include_mask(xs), Eigen::placeholders::all); }
std::vector<Index> Set::include_mask(ConstMatrixRef xs) const {
  LUCID_CHECK_ARGUMENT_EQ(xs.cols(), dimension());
  std::vector<Index> indices;
  indices.reserve(xs.rows());
  for (Index i = 0; i < xs.rows(); i++) {
    if (contains(xs.row(i))) indices.push_back(i);
  }
  return indices;
}

Matrix Set::exclude(ConstMatrixRef xs) const { return xs(exclude_mask(xs), Eigen::placeholders::all); }
std::vector<Index> Set::exclude_mask(ConstMatrixRef xs) const {
  LUCID_CHECK_ARGUMENT_EQ(xs.cols(), dimension());
  std::vector<Index> indices;
  indices.reserve(xs.rows());
  for (Index i = 0; i < xs.rows(); i++) {
    if (!contains(xs.row(i))) indices.push_back(i);
  }
  return indices;
}

void Set::change_size(const double delta_size) { change_size(Vector::Constant(dimension(), delta_size)); }
void Set::change_size(ConstVectorRef) { LUCID_NOT_IMPLEMENTED(); }
Matrix Set::lattice(const Index points_per_dim, const bool endpoint) const {
  return lattice(VectorI::Constant(dimension(), points_per_dim), endpoint);
}

std::unique_ptr<RectSet> Set::to_rect_set() const { LUCID_NOT_IMPLEMENTED(); }

std::ostream& operator<<(std::ostream& os, const Set& set) {
  if (const auto* casted_set = dynamic_cast<const RectSet*>(&set)) return os << *casted_set;
  if (const auto* casted_set = dynamic_cast<const MultiSet*>(&set)) return os << *casted_set;
  if (const auto* casted_set = dynamic_cast<const SphereSet*>(&set)) return os << *casted_set;
  if (const auto* casted_set = dynamic_cast<const PolytopeSet*>(&set)) return os << *casted_set;
  return os << "Set( )";
}

}  // namespace lucid
