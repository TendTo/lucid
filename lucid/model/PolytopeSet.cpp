/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * PolytopeSet class.
 */
#include "lucid/model/PolytopeSet.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "lucid/lib/eigen.h"
#include "lucid/model/RectSet.h"
#include "lucid/util/error.h"
#include "lucid/verification/AlglibOptimiser.h"

namespace lucid {

namespace {

std::uniform_real_distribution<> dis(0.0, 1.0);

Matrix initializer_list_to_matrix(const std::initializer_list<std::initializer_list<Scalar>> list) {
  if (list.size() == 0) return Matrix{};
  const std::size_t num_cols = list.begin()->size();
  Matrix m(list.size(), num_cols);
  Index row = 0;
  for (const auto& row_list : list) {
    if (row_list.size() != num_cols) LUCID_INVALID_ARGUMENT("matrix rows", "must have the same number of columns");
    std::ranges::copy(row_list, m.row(row).data());
    ++row;
  }
  return m;
}

Vector initializer_list_to_vector(std::initializer_list<Scalar> list) {
  Vector v(list.size());
  std::ranges::copy(list, v.data());
  return v;
}

}  // namespace

PolytopeSet::PolytopeSet(Matrix A, Vector b) : A_{std::move(A)}, b_{std::move(b)} {
  LUCID_CHECK_ARGUMENT_EQ(A_.rows(), b_.size());
  LUCID_CHECK_ARGUMENT_CMP(A_.rows(), >, 0);
  LUCID_CHECK_ARGUMENT_CMP(A_.cols(), >, 0);
}

PolytopeSet::PolytopeSet(std::initializer_list<std::initializer_list<Scalar>> A, std::initializer_list<Scalar> b)
    : PolytopeSet{initializer_list_to_matrix(A), initializer_list_to_vector(b)} {}

PolytopeSet PolytopeSet::from_box(const std::vector<std::pair<Scalar, Scalar>>& bounds) {
  LUCID_CHECK_ARGUMENT_CMP(bounds.size(), >, 0);

  const Dimension n = bounds.size();
  Matrix A(2 * n, n);
  Vector b(2 * n);

  // For each dimension i, add constraints:
  // x_i <= ub_i  (represented as [0,...,1,...,0] * x <= ub_i)
  // x_i >= lb_i  (represented as [0,...,-1,...,0] * x <= -lb_i)
  for (Dimension i = 0; i < n; ++i) {
    const auto& [lb, ub] = bounds[i];
    if (lb > ub) {
      LUCID_INVALID_ARGUMENT("bounds", "lower bound must be <= upper bound");
    }

    // Upper bound constraint: x_i <= ub_i
    A.row(i).setZero();
    A(i, i) = 1.0;
    b(i) = ub;

    // Lower bound constraint: -x_i <= -lb_i (i.e., x_i >= lb_i)
    A.row(n + i).setZero();
    A(n + i, i) = -1.0;
    b(n + i) = -lb;
  }

  return PolytopeSet{std::move(A), std::move(b)};
}

bool PolytopeSet::operator()(ConstVectorRef x) const {
  LUCID_CHECK_ARGUMENT_EQ(x.size(), dimension());
  // Check if A*x <= b (with tolerance)
  const Vector result = A_ * x.transpose();
  return (result.array() <= b_.array() + std::numeric_limits<double>::epsilon()).all();
}

const std::pair<Vector, Vector>& PolytopeSet::bounding_box() const {
  if (!bbox_.has_value()) bbox_ = compute_bounding_box();
  return *bbox_;
}

std::pair<Vector, Vector> PolytopeSet::compute_bounding_box() const {
  // For each dimension, solve linear programming problems to find bounds
  LUCID_NOT_SUPPORTED("Bounding");
}

Matrix PolytopeSet::sample(const Index num_samples) const {
  LUCID_NOT_IMPLEMENTED();
  LUCID_CHECK_ARGUMENT_CMP(num_samples, >, 0);
#if 0
  const auto& [lower, upper] = bounding_box();
  Matrix samples(num_samples, dimension());

  // Rejection sampling within the bounding box
  Index samples_generated = 0;
  const Index max_attempts = num_samples * 1000;  // Avoid infinite loops
  Index attempts = 0;

  while (samples_generated < num_samples && attempts < max_attempts) {
    // Generate random point in bounding box
    Vector candidate(dimension());
    for (Dimension i = 0; i < dimension(); ++i) {
      if (std::isfinite(lower(i)) && std::isfinite(upper(i))) {
        candidate(i) = lower(i) + (upper(i) - lower(i)) * dis(random::gen);
      } else {
        // If bounds are infinite, use a reasonable range
        candidate(i) = -10.0 + 20.0 * dis(random::gen);
      }
    }

    // Check if point is in polytope
    if ((*this)(candidate)) {
      samples.row(samples_generated) = candidate;
      ++samples_generated;
    }
    ++attempts;
  }

  if (samples_generated < num_samples) {
    // If we couldn't generate enough samples, fill the rest with the last valid sample
    for (Index i = samples_generated; i < num_samples; ++i) {
      if (samples_generated > 0) {
        samples.row(i) = samples.row(samples_generated - 1);
      } else {
        // If no valid samples, return zeros (this shouldn't happen for valid polytopes)
        samples.row(i).setZero();
      }
    }
  }

  return samples;
#endif
}

Matrix PolytopeSet::lattice(const VectorI& points_per_dim, const bool endpoint) const {
  LUCID_CHECK_ARGUMENT_EQ(points_per_dim.size(), dimension());

  const auto& [lower, upper] = bounding_box();

  const Matrix x_lattice{RectSet{lower, upper}.lattice(points_per_dim, endpoint)};

  // Filter points that are actually in the polytope
  std::vector<Index> valid_indices;
  for (Index i = 0; i < x_lattice.rows(); ++i) {
    if (contains(x_lattice.row(i))) valid_indices.push_back(i);
  }

  return x_lattice(valid_indices, Eigen::placeholders::all);
}

void PolytopeSet::scale(const Scalar factor) { b_ *= factor; }

std::string PolytopeSet::to_string() const { return fmt::format("PolytopeSet( A( {} ) b( {} ) )", A_, b_); }

std::ostream& operator<<(std::ostream& os, const PolytopeSet& set) { return os << set.to_string(); }

}  // namespace lucid
