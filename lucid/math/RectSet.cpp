/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/RectSet.h"

#include <ostream>
#include <random>
#include <string>
#include <utility>

#include "lucid/util/error.h"
#ifdef LUCID_MATPLOTLIB_BUILD
#include "lucid/util/matplotlib.h"
#endif

namespace lucid {

namespace {

std::random_device rd;   // Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> dis(0.0, 1.0);
Vector initializer_list_to_vector(std::initializer_list<Scalar> list) {
  Vector v(list.size());
  std::ranges::copy(list, v.data());
  return v;
}
template <int I, template <class> class T>
Vector bounds_to_vector(const T<std::pair<Scalar, Scalar>>& bounds) {
  Vector v(bounds.size());
  Index i = 0;
  for (const std::pair<Scalar, Scalar> bound : bounds) {
    if constexpr (I == 0) {
      v(i++) = bound.first;
    } else {
      v(i++) = bound.second;
    }
  }
  return v;
}

}  // namespace

RectSet::RectSet(Vector lb, Vector ub, const int seed) : lb_{std::move(lb)}, ub_{std::move(ub)} {
  if (lb_.size() != ub_.size()) LUCID_INVALID_ARGUMENT("lb and ub", "must have the same size");
  if (lb_.size() == 0) LUCID_INVALID_ARGUMENT("lb and ub", "must have at least one element");
  if (seed >= 0) gen.seed(seed);
}
RectSet::RectSet(const std::initializer_list<Scalar> lb, const std::initializer_list<Scalar> ub, const int seed)
    : RectSet{initializer_list_to_vector(lb), initializer_list_to_vector(ub), seed} {}
RectSet::RectSet(std::vector<std::pair<Scalar, Scalar>> bounds, const int seed)
    : RectSet{bounds_to_vector<0, std::vector>(bounds), bounds_to_vector<1, std::vector>(bounds), seed} {}
RectSet::RectSet(std::initializer_list<std::pair<Scalar, Scalar>> bounds, const int seed)
    : RectSet{bounds_to_vector<0>(bounds), bounds_to_vector<1>(bounds), seed} {}

bool RectSet::operator()(ConstMatrixRef x) const {
  if (x.rows() != lb_.rows() || x.cols() != lb_.cols()) {
    LUCID_INVALID_ARGUMENT_EXPECTED("x shape", fmt::format("{} x {}", x.rows(), x.cols()),
                                    fmt::format("{} x {}", lb_.rows(), lb_.cols()));
  }
  return (x.array() >= lb_.array()).all() && (x.array() <= ub_.array()).all();
}

void RectSet::plot(const std::string& color) const {
#ifdef LUCID_MATPLOTLIB_BUILD
  Vector x(lb_.size());
  x << lb_(0), ub_(0);
  Vector y1(1);
  y1 << lb_(1);
  Vector y2(1);
  y2 << ub_(1);
  plt::fill_between(x, y1, y2, {.alpha = 1, .edgecolor = color, .facecolor = "none"});
#else
  LUCID_NOT_SUPPORTED("plot without matplotlib");
#endif
}

void RectSet::plot3d(const std::string&) const { LUCID_NOT_IMPLEMENTED(); }

Matrix RectSet::lattice(const Eigen::VectorX<Index>& points_per_dim, const bool include_endpoints) const {
  if (points_per_dim.size() != lb_.size()) {
    LUCID_INVALID_ARGUMENT_EXPECTED("points_per_dim size", points_per_dim.size(), lb_.size());
  }
  Matrix x_lattice{1, points_per_dim(0)};
  if (include_endpoints) {
    x_lattice.row(0) = Vector::LinSpaced(points_per_dim(0), lb_(0), ub_(0));
    for (Dimension i = 1; i < dimension(); ++i) {
      x_lattice = combvec(x_lattice, Vector::LinSpaced(points_per_dim(i), lb_(i), ub_(i)).transpose());
    }
  } else {
    const Vector delta_per_dim = (ub_ - lb_).cwiseQuotient(points_per_dim.cast<Scalar>());
    x_lattice.row(0) = arange(lb_(0), ub_(0), delta_per_dim(0));
    for (Dimension i = 1; i < dimension(); ++i) {
      x_lattice = combvec(x_lattice, arange(lb_(i), ub_(i), delta_per_dim(i)).transpose());
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

Matrix RectSet::sample_element(const Index num_samples) const {
  Matrix samples(num_samples, dimension());
  const auto diff_vector{ub_ - lb_};
  for (int i = 0; i < num_samples; i++) {
    for (Index j = 0; j < dimension(); j++) {
      samples(i, j) = diff_vector(j) * dis(gen) + lb_(j);
    }
  }
  return samples;
}

std::ostream& operator<<(std::ostream& os, const RectSet& set) {
  return os << fmt::format("RectInterval([{}], [{}])", set.lower_bound().transpose(), set.upper_bound().transpose());
}

}  // namespace lucid
