/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/RectSet.h"

#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "lucid/util/error.h"
#include "lucid/util/random.h"

namespace lucid {

namespace {

std::uniform_real_distribution<> dis(0.0, 1.0);
Vector initializer_list_to_vector(std::initializer_list<Scalar> list) {
  Vector v(list.size());
  std::ranges::copy(list, v.data());
  return v;
}
template <int I, template <class, class...> class T>
Vector bounds_to_vector(const T<std::pair<Scalar, Scalar>>& bounds) {
  Vector v(bounds.size());
  Index i = 0;
  for (const auto& [first, second] : bounds) {
    if constexpr (I == 0) {
      v(i++) = first;
    } else {
      v(i++) = second;
    }
  }
  return v;
}

}  // namespace

RectSet::RectSet(Vector lb, Vector ub) : lb_{std::move(lb)}, ub_{std::move(ub)} {
  if (lb_.size() != ub_.size()) LUCID_INVALID_ARGUMENT("lb and ub", "must have the same size");
  if (lb_.size() == 0) LUCID_INVALID_ARGUMENT("lb and ub", "must have at least one element");
}
RectSet::RectSet(const std::initializer_list<Scalar> lb, const std::initializer_list<Scalar> ub)
    : RectSet{initializer_list_to_vector(lb), initializer_list_to_vector(ub)} {}
RectSet::RectSet(const std::vector<std::pair<Scalar, Scalar>>& bounds)
    : RectSet{bounds_to_vector<0, std::vector>(bounds), bounds_to_vector<1, std::vector>(bounds)} {}
RectSet::RectSet(const std::initializer_list<std::pair<Scalar, Scalar>> bounds)
    : RectSet{bounds_to_vector<0>(bounds), bounds_to_vector<1>(bounds)} {}

bool RectSet::operator()(ConstVectorRef x) const {
  return (x.array() >= lb_.array()).all() && (x.array() <= ub_.array()).all();
}

Matrix RectSet::lattice(const VectorI& points_per_dim, const bool include_endpoints) const {
  if (points_per_dim.size() != lb_.size()) {
    LUCID_INVALID_ARGUMENT_EXPECTED("points_per_dim size", points_per_dim.size(), lb_.size());
  }
  Matrix x_lattice{1, points_per_dim(0)};
  if (include_endpoints) {
    x_lattice.row(0) = Vector::LinSpaced(points_per_dim(0), lb_(0), ub_(0));
    for (Dimension i = 1; i < dimension(); ++i) {
      x_lattice = combvec(x_lattice, Vector::LinSpaced(points_per_dim(i), lb_(i), ub_(i)));
    }
  } else {
    const Vector delta_per_dim = (ub_ - lb_).cwiseQuotient(points_per_dim.cast<Scalar>());
    x_lattice.row(0) = arange(lb_(0), ub_(0), delta_per_dim(0));
    for (Dimension i = 1; i < dimension(); ++i) {
      x_lattice = combvec(x_lattice, arange(lb_(i), ub_(i), delta_per_dim(i)));
    }
  }
  x_lattice.transposeInPlace();
  return x_lattice;
}

RectSet::operator Matrix() const {
  Matrix x_lim{2, lb_.size()};
  x_lim << lb_.transpose(), ub_.transpose();
  return x_lim;
}

Matrix RectSet::sample(const Index num_samples) const {
  Matrix samples(num_samples, dimension());
  const auto diff_vector{ub_ - lb_};
  for (int i = 0; i < num_samples; i++) {
    for (Index j = 0; j < dimension(); j++) {
      samples(i, j) = diff_vector(j) * dis(random::gen) + lb_(j);
    }
  }
  return samples;
}

std::ostream& operator<<(std::ostream& os, const RectSet& set) {
  return os << fmt::format("RectSet( lb( [{}] ) ub( [{}] ) )", set.lower_bound(), set.upper_bound());
}

}  // namespace lucid
