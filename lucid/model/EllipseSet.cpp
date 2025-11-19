/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * EllipseSet class.
 */
#include "lucid/model/EllipseSet.h"

#include <limits>
#include <ostream>
#include <vector>

#include "lucid/model/RectSet.h"
#include "lucid/util/error.h"
#include "lucid/util/random.h"

namespace lucid {

namespace {
std::uniform_real_distribution<> uniform(0.0, 1.0);
std::normal_distribution<> normal(0.0, 1.0);
}  // namespace

EllipseSet::EllipseSet(ConstVectorRef center, ConstVectorRef radii) : center_(center), radii_(radii) {
  LUCID_CHECK_ARGUMENT_CMP(center.size(), >, 0);
  LUCID_CHECK_ARGUMENT_EQ(center.size(), radii.size());
  LUCID_CHECK_ARGUMENT_CMP(radii.minCoeff(), >=, 0);
}

EllipseSet::EllipseSet(ConstVectorRef center, const Scalar radius)
    : EllipseSet{center, Vector::Constant(center.size(), radius)} {}

Matrix EllipseSet::sample(const Index num_samples) const {
  // Generate samples uniformly within the ellipsoid using the method:
  // 1. Generate points uniformly in a unit sphere
  // 2. Scale each dimension by the corresponding radius
  // 3. Translate to the center

  Matrix u{Matrix::NullaryExpr(num_samples, dimension(), [](const Index, const Index) { return normal(random::gen); })};
  const Vector norm{u.rowwise().norm()};

  // Generate random radii using the d-th root to ensure uniform distribution
  const Matrix r{Matrix::NullaryExpr(num_samples, 1, [](const Index, const Index) { return uniform(random::gen); })
                     .array()
                     .pow(1.0 / dimension())};

  // Normalize to unit sphere, scale by random radius, then scale by ellipsoid radii
  Matrix samples = (u.array().colwise() / norm.transpose().array()).rowwise() * radii_.array();
  samples = (samples.array().colwise() * r.col(0).array()).rowwise() + center_.array();

  return samples;
}

bool EllipseSet::operator()(ConstVectorRef x) const {
  LUCID_CHECK_ARGUMENT_EQ(x.size(), center_.size());

  // Check if the vector is in the ellipsoid set
  // Sum of squared normalized distances should be <= 1
  const Vector normalized = (x - center_).cwiseQuotient(radii_);
  return normalized.squaredNorm() <= 1.0 + std::numeric_limits<Scalar>::epsilon();
}

Matrix EllipseSet::lattice(const VectorI& points_per_dim, const bool endpoint) const {
  // Generate a lattice by creating a bounding box and filtering points
  const RectSet rect_set{center_ - radii_, center_ + radii_};
  const Matrix lattice{rect_set.lattice(points_per_dim, endpoint)};

  std::vector<Index> mask_rows;
  mask_rows.reserve(lattice.rows());
  for (Index i = 0; i < lattice.rows(); ++i) {
    if (this->contains(lattice.row(i))) mask_rows.push_back(i);
  }

  return lattice(mask_rows, Eigen::placeholders::all);
}

Vector EllipseSet::general_lower_bound() const { return center_ - radii_; }
Vector EllipseSet::general_upper_bound() const { return center_ + radii_; }

void EllipseSet::change_size(ConstVectorRef delta_size) {
  LUCID_TRACE_FMT("({})", LUCID_FORMAT_MATRIX(delta_size));
  LUCID_CHECK_ARGUMENT_EQ(delta_size.size(), dimension());

  // Update each radius by half of the delta (since delta affects diameter)
  const Vector new_radii = radii_ + delta_size / 2.0;
  LUCID_CHECK_ARGUMENT_CMP(new_radii.minCoeff(), >=, 0);

  radii_ = new_radii;

  LUCID_TRACE_FMT("=> {}", *this);
}

std::unique_ptr<Set> EllipseSet::to_rect_set() const {
  // Convert to the smallest axis-aligned bounding box
  return std::make_unique<RectSet>(center_ - radii_, center_ + radii_);
}

bool EllipseSet::operator==(const EllipseSet& other) const {
  return center_.isApprox(other.center_) && radii_.isApprox(other.radii_);
}

bool EllipseSet::operator==(const Set& other) const {
  if (Set::operator==(other)) return true;
  if (const auto* other_ellipse = dynamic_cast<const EllipseSet*>(&other)) return *this == *other_ellipse;
  return false;
}

std::string EllipseSet::to_string() const {
  return fmt::format("EllipseSet( center( [{}] ) radii( [{}] ) )", center_, radii_);
}

std::ostream& operator<<(std::ostream& os, const EllipseSet& set) { return os << set.to_string(); }

}  // namespace lucid
