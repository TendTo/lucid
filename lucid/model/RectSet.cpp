/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/RectSet.h"

#include <algorithm>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "lucid/model/MultiSet.h"
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

Matrix RectSet::lattice(const VectorI& points_per_dim, const bool endpoint) const {
  if (points_per_dim.size() != lb_.size()) {
    LUCID_INVALID_ARGUMENT_EXPECTED("points_per_dim size", points_per_dim.size(), lb_.size());
  }
  const int add_point = endpoint ? 0 : 1;
  Matrix x_lattice{1, points_per_dim(0)};
  x_lattice.row(0) = Vector::LinSpaced(points_per_dim(0) + add_point, lb_(0), ub_(0)).head(points_per_dim(0));
  for (Dimension i = 1; i < dimension(); ++i) {
    x_lattice =
        combvec(x_lattice, Vector::LinSpaced(points_per_dim(i) + add_point, lb_(i), ub_(i)).head(points_per_dim(i)));
  }
  x_lattice.transposeInPlace();
  return x_lattice;
}

void RectSet::change_size(ConstVectorRef delta_size) {
  LUCID_TRACE_FMT("({})", LUCID_FORMAT_MATRIX(delta_size));
  LUCID_CHECK_ARGUMENT_EQ(delta_size.size(), dimension());

  const Vector center = (lb_ + ub_) / 2.0;
  const Vector half_size = (ub_ - lb_) / 2.0;

  // Compute the new half-size (adding half of delta_size to each side)
  const Vector new_half_size = half_size + delta_size / 2.0;
  LUCID_CHECK_ARGUMENT_CMP(new_half_size.minCoeff(), >=, 0);

  // Update bounds to maintain the center
  lb_ = center - new_half_size;
  ub_ = center + new_half_size;

  LUCID_TRACE_FMT("=> {}", *this);
}
RectSet RectSet::relative_to(const RectSet& set) const { return relative_to(set.lower_bound()); }
RectSet RectSet::relative_to(ConstVectorRef point) const {
  LUCID_CHECK_ARGUMENT_EQ(point.size(), dimension());
  return RectSet{lb_ - point, ub_ - point};
}

RectSet RectSet::scale(ConstVectorRef scale, const RectSet& bounds, const bool relative_to_bounds) const {
  LUCID_CHECK_ARGUMENT_EQ(dimension(), scale.size());
  LUCID_CHECK_ARGUMENT_CMP(scale.minCoeff(), >, 0);
  RectSet result{*this};
  const Vector size_change =
      (relative_to_bounds ? bounds.sizes() : sizes()).array() * (scale.array() - (relative_to_bounds ? 0 : 1)) / 2.0;
  result.lb_ = ((lb_ - size_change).array() < bounds.lb_.array()).select(bounds.lb_, lb_ - size_change);
  result.ub_ = ((ub_ + size_change).array() > bounds.ub_.array()).select(bounds.ub_, ub_ + size_change);
  return result;
}

RectSet RectSet::scale(const double scale, const RectSet& bounds, const bool relative_to_bounds) const {
  return this->scale(Vector::Constant(dimension(), scale), bounds, relative_to_bounds);
}
std::unique_ptr<Set> RectSet::scale_wrapped_impl(ConstVectorRef scale, const RectSet& bounds,
                                                 const bool relative_to_bounds) const {
  LUCID_CHECK_ARGUMENT_EQ(dimension(), scale.size());
  LUCID_CHECK_ARGUMENT_CMP(scale.minCoeff(), >=, 0);

  std::vector<std::unique_ptr<Set>> sets;
  sets.emplace_back(std::make_unique<RectSet>(*this));

  // First, add the original set, scaled but bounded
  RectSet& bounded = *static_cast<RectSet*>(sets.back().get());
  const Vector size_change =
      (relative_to_bounds ? bounds.sizes() : sizes()).array() * (scale.array() - (relative_to_bounds ? 0 : 1)) / 2.0;
  Vector new_lb{lb_ - size_change};
  Vector new_ub{ub_ + size_change};
  // Ensure we haven't removed or inverted any dimensions
  LUCID_CHECK_ARGUMENT_CMP((new_ub - new_lb).minCoeff(), >, 0);
  bounded.lb_ = (new_lb.array() < bounds.lb_.array()).select(bounds.lb_, new_lb);
  bounded.ub_ = (new_ub.array() > bounds.ub_.array()).select(bounds.ub_, new_ub);

  // Then, for each dimension, check if wrapping is needed
  for (Dimension i = 0; i < dimension(); ++i) {
    // Wrapping on lower bound. No point in doing it if there is no space
    if (new_lb(i) < bounds.lb_(i) && bounded.ub_(i) < bounds.ub_(i)) {
      sets.emplace_back(std::make_unique<RectSet>(bounded));
      RectSet& wrap_set = *static_cast<RectSet*>(sets.back().get());
      wrap_set.lb_(i) = std::max(bounds.ub_(i) - (bounds.lb_(i) - new_lb(i)), bounded.ub_(i));  // Avoid overlap
      wrap_set.ub_(i) = bounds.ub_(i);
    }
    // Wrapping on upper bound. No point in doing it if there is no space
    if (new_ub(i) > bounds.ub_(i) && bounded.lb_(i) > bounds.lb_(i)) {
      sets.emplace_back(std::make_unique<RectSet>(bounded));
      RectSet& wrap_set = *static_cast<RectSet*>(sets.back().get());
      wrap_set.lb_(i) = bounds.lb_(i);
      wrap_set.ub_(i) = std::min(bounds.lb_(i) + (new_ub(i) - bounds.ub_(i)), bounded.lb_(i));  // Avoid overlap
    }
  }
  if (sets.size() == 1) return std::move(sets.back());
  return std::make_unique<MultiSet>(std::move(sets));
}

void RectSet::increase_size_impl(ConstVectorRef size_increase) {
  lb_ -= size_increase / 2.0;
  ub_ += size_increase / 2.0;
}

std::unique_ptr<Set> RectSet::to_rect_set() const { return std::make_unique<RectSet>(*this); }

RectSet& RectSet::operator+=(ConstVectorRef offset) {
  LUCID_CHECK_ARGUMENT_EQ(dimension(), offset.size());
  lb_ += offset;
  ub_ += offset;
  return *this;
}
RectSet& RectSet::operator-=(ConstVectorRef offset) {
  LUCID_CHECK_ARGUMENT_EQ(dimension(), offset.size());
  lb_ -= offset;
  ub_ -= offset;
  return *this;
}
RectSet RectSet::scale(ConstVectorRef scale) const {
  RectSet result{*this};
  const Vector size_change = sizes().array() * scale.array() / 2.0;
  result.lb_ = lb_ - size_change;
  result.ub_ = ub_ + size_change;
  return result;
}
RectSet RectSet::scale(const double scale) const { return this->scale(Vector::Constant(dimension(), scale)); }

RectSet& RectSet::operator*=(ConstVectorRef scale) {
  LUCID_CHECK_ARGUMENT_EQ(dimension(), scale.size());
  LUCID_CHECK_ARGUMENT_CMP(scale.minCoeff(), >, 0);
  lb_ = lb_.cwiseProduct(scale);
  ub_ = ub_.cwiseProduct(scale);
  return *this;
}
RectSet& RectSet::operator/=(ConstVectorRef scale) {
  LUCID_CHECK_ARGUMENT_EQ(dimension(), scale.size());
  LUCID_CHECK_ARGUMENT_CMP(scale.minCoeff(), >, 0);
  lb_ = lb_.cwiseQuotient(scale);
  ub_ = ub_.cwiseQuotient(scale);
  return *this;
}
bool RectSet::operator==(const Set& other) const {
  if (Set::operator==(other)) return true;
  if (const auto other_rect = dynamic_cast<const RectSet*>(&other)) return *this == *other_rect;
  return false;
}
bool RectSet::operator==(const RectSet& other) const {
  return dimension() == other.dimension() && lb_ == other.lb_ && ub_ == other.ub_;
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

std::string RectSet::to_string() const { return fmt::format("RectSet( lb( [{}] ) ub( [{}] ) )", lb_, ub_); }

std::unique_ptr<Set> RectSet::clone() const { return std::make_unique<RectSet>(*this); }

std::ostream& operator<<(std::ostream& os, const RectSet& set) { return os << set.to_string(); }

}  // namespace lucid
