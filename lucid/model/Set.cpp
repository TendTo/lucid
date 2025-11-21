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

#include "lucid/model/RectSet.h"
#include "lucid/util/IndexIterator.h"
#include "lucid/util/error.h"

namespace lucid {

namespace {

bool contains_wrapped_batch(const Set& set, IndexIterator<std::vector<Index>>& it, ConstMatrixRef xs,
                            ConstVectorRef period, const Index i) {
  bool contained = false;
  for (it.reset(); it && !contained; ++it) {  // Check all wrapped copies of that point to see if any is in the set
    Vector wrapped_x = xs.row(i);
    for (Index d = 0; d < set.dimension(); d++) {
      wrapped_x(d) += static_cast<double>(it[d]) * period(d);
    }
    if (set.contains(wrapped_x)) contained = true;  // If any wrapped copy is in the set, mark it as contained
  }
  return contained;
}

IndexIterator<std::vector<Index>> period_wrapping_index_iterator(const Set& set, ConstVectorRef period) {
  std::vector<Index> num_periods_below(set.dimension());
  std::vector<Index> num_periods_above(set.dimension());
  const Vector glb{set.general_lower_bound().cwiseQuotient(period)};
  const Vector gub{set.general_upper_bound().cwiseQuotient(period)};
  for (Index d = 0; d < set.dimension(); d++) {
    num_periods_below[d] = static_cast<Index>(std::floor(glb(d)));
    num_periods_above[d] = static_cast<Index>(std::ceil(gub(d)));
  }
  return {num_periods_below, num_periods_above};
}

}  // namespace

Vector Set::sample() const { return sample(1l).row(0); }

bool Set::contains_wrapped(ConstVectorRef x, ConstVectorRef period, const Dimension num_periods) const {
  return contains_wrapped(x, period, std::vector<Dimension>(dimension(), num_periods));
}
bool Set::contains_wrapped(ConstVectorRef x, ConstVectorRef period, const std::vector<Dimension>& num_periods) const {
  return contains_wrapped(x, period, num_periods, num_periods);
}
bool Set::contains_wrapped(ConstVectorRef x, ConstVectorRef period, const Dimension num_periods_below,
                           const Dimension num_periods_above) const {
  return contains_wrapped(x, period, std::vector<Dimension>(dimension(), num_periods_below),
                          std::vector<Dimension>(dimension(), num_periods_above));
}
bool Set::contains_wrapped(ConstVectorRef x, ConstVectorRef period, const std::vector<Dimension>& num_periods_below,
                           const std::vector<Dimension>& num_periods_above) const {
  LUCID_CHECK_ARGUMENT_EQ(x.size(), dimension());             // x must have the same dimension as the set
  LUCID_CHECK_ARGUMENT_EQ(period.size(), dimension());        // period must have the same dimension as the set
  LUCID_CHECK_ARGUMENT_CMP(period.minCoeff(), >, 0.0);        // period must be positive
  LUCID_CHECK_ARGUMENT_CMP(x.minCoeff(), >=, 0.0);            // x must be in [0, period)
  LUCID_CHECK_ARGUMENT_CMP((x - period).maxCoeff(), <, 0.0);  // x must be in [0, period)
  LUCID_CHECK_ARGUMENT_EQ(num_periods_below.size(), static_cast<std::size_t>(dimension()));
  LUCID_CHECK_ARGUMENT_EQ(num_periods_above.size(), static_cast<std::size_t>(dimension()));
  LUCID_CHECK_ARGUMENT_CMP(std::ranges::min(num_periods_above), >=, 0);  // num_periods_above must be non-negative
  LUCID_CHECK_ARGUMENT_CMP(std::ranges::min(num_periods_below), >=, 0);  // num_periods_below must be non-negative

  std::vector<Index> max_values(dimension());
  for (Index d = 0; d < dimension(); d++) {
    max_values[d] = num_periods_below[d] + num_periods_above[d] + 1;  // +1 to include 0 wrapping
  }
  for (IndexIterator it{max_values}; it; ++it) {
    Vector wrapped_x = x;
    for (Index d = 0; d < dimension(); d++) {
      // Determine the wrapping coefficient: <num_periods => -period, =num_periods => no wrap, ><num_periods >= +period
      const double coeff = static_cast<double>(it[d] - num_periods_below[d]);
      wrapped_x(d) += coeff * period(d);
    }
    if (contains(wrapped_x)) return true;
  }
  return false;
}
bool Set::contains_wrapped(ConstVectorRef x, ConstVectorRef period) const {
  LUCID_CHECK_ARGUMENT_EQ(x.size(), dimension());             // x must have the same dimension as the set
  LUCID_CHECK_ARGUMENT_EQ(period.size(), dimension());        // period must have the same dimension as the set
  LUCID_CHECK_ARGUMENT_CMP(period.minCoeff(), >, 0.0);        // period must be positive
  LUCID_CHECK_ARGUMENT_CMP(x.minCoeff(), >=, 0.0);            // x must be in [0, period)
  LUCID_CHECK_ARGUMENT_CMP((x - period).maxCoeff(), <, 0.0);  // x must be in [0, period)

  std::vector<Index> num_periods_below(dimension());
  std::vector<Index> num_periods_above(dimension());
  const Vector glb{general_lower_bound().cwiseQuotient(period)};
  const Vector gub{general_upper_bound().cwiseQuotient(period)};
  for (Index d = 0; d < dimension(); d++) {
    num_periods_below[d] = static_cast<Index>(std::floor(glb(d)));
    num_periods_above[d] = static_cast<Index>(std::ceil(gub(d)));
  }
  for (IndexIterator it{num_periods_below, num_periods_above}; it; ++it) {
    Vector wrapped_x = x;
    for (Index d = 0; d < dimension(); d++) {
      wrapped_x(d) += static_cast<double>(it[d]) * period(d);
    }
    if (contains(wrapped_x)) return true;
  }
  return false;
}

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
std::vector<Index> Set::include_mask_wrapped(ConstMatrixRef xs, ConstVectorRef period) const {
  LUCID_CHECK_ARGUMENT_EQ(xs.cols(), dimension());
  LUCID_CHECK_ARGUMENT_EQ(period.size(), dimension());  // period must have the same dimension as the set
  LUCID_CHECK_ARGUMENT_CMP(period.minCoeff(), >, 0.0);  // period must be positive
  LUCID_CHECK_ARGUMENT_CMP(xs.minCoeff(), >=, 0.0);     // x must be in [0, period)
  LUCID_CHECK_ARGUMENT_CMP((xs.rowwise() - period).maxCoeff(), <, 0.0);  // x must be in [0, period)

  std::vector<Index> indices;
  indices.reserve(xs.rows());
  IndexIterator it{period_wrapping_index_iterator(*this, period)};
  for (Index i = 0; i < xs.rows(); i++) {
    if (contains_wrapped_batch(*this, it, xs, period, i)) indices.push_back(i);
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

std::vector<Index> Set::exclude_mask_wrapped(ConstMatrixRef xs, ConstVectorRef period) const {
  LUCID_CHECK_ARGUMENT_EQ(xs.cols(), dimension());
  LUCID_CHECK_ARGUMENT_EQ(period.size(), dimension());  // period must have the same dimension as the set
  LUCID_CHECK_ARGUMENT_CMP(period.minCoeff(), >, 0.0);  // period must be positive
  LUCID_CHECK_ARGUMENT_CMP(xs.minCoeff(), >=, 0.0);     // x must be in [0, period)
  LUCID_CHECK_ARGUMENT_CMP((xs.rowwise() - period).maxCoeff(), <, 0.0);  // x must be in [0, period)

  std::vector<Index> indices;
  indices.reserve(xs.rows());
  IndexIterator it{period_wrapping_index_iterator(*this, period)};
  for (Index i = 0; i < xs.rows(); i++) {
    if (!contains_wrapped_batch(*this, it, xs, period, i)) indices.push_back(i);
  }
  return indices;
}

std::pair<std::vector<Index>, std::vector<Index>> Set::include_exclude_masks(ConstMatrixRef xs) const {
  LUCID_CHECK_ARGUMENT_EQ(xs.cols(), dimension());
  std::pair<std::vector<Index>, std::vector<Index>> masks;
  masks.first.reserve(xs.rows());
  masks.second.reserve(xs.rows());
  for (Index i = 0; i < xs.rows(); i++) {  // For each point in xs
    if (contains(xs.row(i)))               // If contained, add to include mask
      masks.first.push_back(i);
    else  // If not contained, add to exclude mask
      masks.second.push_back(i);
  }
  return masks;
}
std::pair<std::vector<Index>, std::vector<Index>> Set::include_exclude_masks_wrapped(ConstMatrixRef xs,
                                                                                     ConstVectorRef period) const {
  LUCID_CHECK_ARGUMENT_EQ(xs.cols(), dimension());
  LUCID_CHECK_ARGUMENT_EQ(period.size(), dimension());  // period must have the same dimension as the set
  LUCID_CHECK_ARGUMENT_CMP(period.minCoeff(), >, 0.0);  // period must be positive
  LUCID_CHECK_ARGUMENT_CMP(xs.minCoeff(), >=, 0.0);     // x must be in [0, period)
  LUCID_CHECK_ARGUMENT_CMP((xs.rowwise() - period).maxCoeff(), <, 0.0);  // x must be in [0, period)

  std::pair<std::vector<Index>, std::vector<Index>> masks;
  masks.first.reserve(xs.rows());
  masks.second.reserve(xs.rows());

  IndexIterator it{period_wrapping_index_iterator(*this, period)};
  for (Index i = 0; i < xs.rows(); i++) {                  // For each point in xs
    if (contains_wrapped_batch(*this, it, xs, period, i))  // If contained, add to include mask
      masks.first.push_back(i);
    else  // If not contained, add to exclude mask
      masks.second.push_back(i);
  }
  return masks;
}
std::pair<std::vector<Index>, std::vector<Index>> Set::include_exclude_masks_wrapped(ConstMatrixRef xs,
                                                                                     const RectSet& period) const {
  LUCID_CHECK_ARGUMENT_EQ(xs.cols(), dimension());
  LUCID_CHECK_ARGUMENT_EQ(period.dimension(), dimension());  // period must have the same dimension as the set

  const Vector period_sizes = period.sizes();
  std::pair<std::vector<Index>, std::vector<Index>> masks;
  masks.first.reserve(xs.rows());
  masks.second.reserve(xs.rows());

  std::vector<Index> num_periods_below(dimension());
  std::vector<Index> num_periods_above(dimension());
  const Vector glb{(general_lower_bound() - period.lower_bound()).cwiseQuotient(period_sizes)};
  const Vector gub{(general_upper_bound() - period.lower_bound()).cwiseQuotient(period_sizes)};
  for (Index d = 0; d < dimension(); d++) {
    num_periods_below[d] = static_cast<Index>(std::floor(glb(d)));
    num_periods_above[d] = static_cast<Index>(std::ceil(gub(d)));
  }
  IndexIterator it{num_periods_below, num_periods_above};
  for (Index i = 0; i < xs.rows(); i++) {                        // For each point in xs
    if (contains_wrapped_batch(*this, it, xs, period_sizes, i))  // If contained, add to include mask
      masks.first.push_back(i);
    else  // If not contained, add to exclude mask
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

template <bool Inplace>
  requires(!Inplace)
std::unique_ptr<Set> Set::increase_size(ConstVectorRef size_increase) const {
  std::unique_ptr<Set> new_set{all_equal(size_increase) ? clone() : to_anisotropic()};
  new_set->increase_size<true>(size_increase);
  return new_set;
}

template <bool Inplace>
  requires(Inplace)
void Set::increase_size(ConstVectorRef size_increase) {
  LUCID_CHECK_ARGUMENT_EQ(size_increase.size(), dimension());
  LUCID_CHECK_ARGUMENT_CMP(size_increase.minCoeff(), >=, 0);
  increase_size_impl(size_increase);
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
Set& Set::operator+=(ConstVectorRef) { LUCID_NOT_IMPLEMENTED(); }
Set& Set::operator+=(const Scalar offset) { return operator+=(Vector::Constant(dimension(), offset)); }
Set& Set::operator-=(ConstVectorRef) { LUCID_NOT_IMPLEMENTED(); }
Set& Set::operator-=(const Scalar offset) { return operator-=(Vector::Constant(dimension(), offset)); }
Set& Set::operator*=(ConstVectorRef) { LUCID_NOT_IMPLEMENTED(); }
Set& Set::operator*=(const Scalar scale) { return operator*=(Vector::Constant(dimension(), scale)); }
Set& Set::operator/=(ConstVectorRef) { LUCID_NOT_IMPLEMENTED(); }
Set& Set::operator/=(const Scalar scale) { return operator/=(Vector::Constant(dimension(), scale)); }

std::unique_ptr<Set> Set::scale_wrapped_impl(ConstVectorRef, const RectSet&, bool) const { LUCID_NOT_IMPLEMENTED(); }
void Set::increase_size_impl(ConstVectorRef) { LUCID_NOT_IMPLEMENTED(); }

std::unique_ptr<Set> Set::to_anisotropic() const { return clone(); }

std::string Set::to_string() const { return "Set( )"; }

std::ostream& operator<<(std::ostream& os, const Set& set) { return os << set.to_string(); }

template std::unique_ptr<Set> Set::increase_size<false>(ConstVectorRef size_increase) const;
template void Set::increase_size<true>(ConstVectorRef size_increase);

}  // namespace lucid
