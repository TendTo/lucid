/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Set.h"

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

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

std::pair<std::vector<Index>, std::vector<Index>> Set::include_exclude_masks(ConstMatrixRef xs) const {
  LUCID_CHECK_ARGUMENT_EQ(xs.cols(), dimension());
  std::pair<std::vector<Index>, std::vector<Index>> masks;
  masks.first.reserve(xs.rows());
  masks.second.reserve(xs.rows());
  for (Index i = 0; i < xs.rows(); i++) {
    if (contains(xs.row(i)))
      masks.first.push_back(i);
    else
      masks.second.push_back(i);
  }
  return masks;
}

std::unique_ptr<Set> Set::scale_wrapped(const double scale, const RectSet& bounds,
                                        const bool relative_to_bounds) const {
  return scale_wrapped(Vector::Constant(dimension(), scale), bounds, relative_to_bounds);
}
std::unique_ptr<Set> Set::scale_wrapped(ConstVectorRef scale, const RectSet& bounds,
                                        const bool relative_to_bounds) const {
  return scale_wrapped_impl(scale, bounds, relative_to_bounds);
}

void Set::change_size(const double delta_size) { change_size(Vector::Constant(dimension(), delta_size)); }
void Set::change_size(ConstVectorRef) { LUCID_NOT_IMPLEMENTED(); }
Matrix Set::lattice(const Index points_per_dim, const bool endpoint) const {
  return lattice(VectorI::Constant(dimension(), points_per_dim), endpoint);
}

std::unique_ptr<Set> Set::to_rect_set() const { LUCID_NOT_IMPLEMENTED(); }
Vector Set::general_lower_bound() const { LUCID_NOT_IMPLEMENTED(); }
Vector Set::general_upper_bound() const { LUCID_NOT_IMPLEMENTED(); }

bool Set::operator==(const Set& other) const { return this == &other; }

std::unique_ptr<Set> Set::scale_wrapped_impl(ConstVectorRef, const RectSet&, bool) const { LUCID_NOT_IMPLEMENTED(); }

std::string Set::to_string() const { return "Set( )"; }

std::ostream& operator<<(std::ostream& os, const Set& set) { return os << set.to_string(); }

}  // namespace lucid
