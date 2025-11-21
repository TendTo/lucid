/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * EllipseSet class.
 */
#include "lucid/model/EllipseSet.h"

#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "lucid/model/RectSet.h"
#include "lucid/model/SphereSet.h"
#include "lucid/util/error.h"
#include "lucid/util/random.h"

namespace lucid {

namespace {
std::uniform_real_distribution<> uniform(0.0, 1.0);
std::normal_distribution<> normal(0.0, 1.0);
}  // namespace

EllipseSet::EllipseSet(ConstVectorRef center, ConstVectorRef semi_axes) : center_(center), semi_axes_(semi_axes) {
  LUCID_CHECK_ARGUMENT_CMP(center.size(), >, 0);
  LUCID_CHECK_ARGUMENT_EQ(center.size(), semi_axes.size());
  LUCID_CHECK_ARGUMENT_CMP(semi_axes.minCoeff(), >, 0);
}

EllipseSet::EllipseSet(ConstVectorRef center, const Scalar radius)
    : EllipseSet{center, Vector::Constant(center.size(), radius)} {}

Matrix EllipseSet::sample(const Index num_samples) const {
  Matrix u{Matrix::NullaryExpr(num_samples, dimension(), [](const Index, const Index) { return normal(random::gen); })};
  const Vector norm{u.rowwise().norm()};
  const Matrix r{
      Matrix::NullaryExpr(num_samples, dimension(), [](const Index, const Index) { return uniform(random::gen); })
          .array()
          .pow(1.0 / dimension())};
  return ((r * semi_axes_.asDiagonal()).cwiseProduct(u).array().colwise() / norm.transpose().array()).rowwise() +
         center_.array();
}

bool EllipseSet::operator()(ConstVectorRef x) const {
  LUCID_CHECK_ARGUMENT_EQ(x.size(), center_.size());
  return (x - center_).cwiseQuotient(semi_axes_).squaredNorm() <= 1.0 + std::numeric_limits<Scalar>::epsilon();
}

Matrix EllipseSet::lattice(const VectorI& points_per_dim, const bool endpoint) const {
  // Generate a lattice by creating a bounding box and filtering points
  const RectSet rect_set{center_ - semi_axes_, center_ + semi_axes_};
  const Matrix lattice{rect_set.lattice(points_per_dim, endpoint)};

  std::vector<Index> mask_rows;
  mask_rows.reserve(lattice.rows());
  for (Index i = 0; i < lattice.rows(); ++i) {
    if (this->contains(lattice.row(i))) mask_rows.push_back(i);
  }

  return lattice(mask_rows, Eigen::placeholders::all);
}

Vector EllipseSet::general_lower_bound() const { return center_ - semi_axes_; }
Vector EllipseSet::general_upper_bound() const { return center_ + semi_axes_; }

void EllipseSet::change_size(ConstVectorRef delta_size) {
  LUCID_TRACE_FMT("({})", LUCID_FORMAT_MATRIX(delta_size));
  LUCID_CHECK_ARGUMENT_EQ(delta_size.size(), dimension());
  // Update each radius by half of the delta (since delta affects diameter)
  const Vector new_semi_axes = semi_axes_ + delta_size / 2.0;
  LUCID_CHECK_ARGUMENT_CMP(new_semi_axes.minCoeff(), >, 0);
  semi_axes_ = new_semi_axes;
  LUCID_TRACE_FMT("=> {}", *this);
}

std::unique_ptr<Set> EllipseSet::to_rect_set() const {
  return std::make_unique<RectSet>(center_ - semi_axes_, center_ + semi_axes_);
}

bool EllipseSet::operator==(const EllipseSet& other) const {
  return dimension() == other.dimension() && center_ == other.center_ && semi_axes_ == other.semi_axes_;
}

bool EllipseSet::operator==(const Set& other) const {
  if (Set::operator==(other)) return true;
  if (const auto* other_ellipse = dynamic_cast<const EllipseSet*>(&other)) return *this == *other_ellipse;
  return false;
}

std::string EllipseSet::to_string() const {
  return fmt::format("EllipseSet( center( [{}] ) semi_axes( [{}] ) )", center_, semi_axes_);
}
std::unique_ptr<Set> EllipseSet::increase_size_impl(ConstVectorRef size_increase) const {
  const Vector new_semi_axes = semi_axes_ + size_increase / 2.0;
  const double new_semi_axes_0 = new_semi_axes(0);
  if (std::ranges::all_of(new_semi_axes, [new_semi_axes_0](auto x) { return x == new_semi_axes_0; })) {
    return std::make_unique<SphereSet>(center_, new_semi_axes_0);
  }
  return std::make_unique<SphereSet>(center_, new_semi_axes_0);
}

std::ostream& operator<<(std::ostream& os, const EllipseSet& set) { return os << set.to_string(); }

}  // namespace lucid
