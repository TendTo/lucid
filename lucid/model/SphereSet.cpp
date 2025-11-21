/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * SphereSet class.
 */
#include "lucid/model/SphereSet.h"

#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "lucid/model/EllipseSet.h"
#include "lucid/model/RectSet.h"
#include "lucid/util/error.h"
#include "lucid/util/random.h"

namespace lucid {

namespace {
std::uniform_real_distribution<> uniform(0.0, 1.0);
std::normal_distribution<> normal(0.0, 1.0);
}  // namespace

SphereSet::SphereSet(ConstVectorRef center, Scalar radius) : center_(center), radius_(radius) {
  LUCID_CHECK_ARGUMENT_CMP(center.size(), >, 0);
  LUCID_CHECK_ARGUMENT_CMP(radius, >=, 0);
}
Matrix SphereSet::sample(const Index num_samples) const {
  Matrix u{Matrix::NullaryExpr(num_samples, dimension(), [](const Index, const Index) { return normal(random::gen); })};
  const Vector norm{u.rowwise().norm()};
  const Matrix r{radius_ * Matrix::NullaryExpr(num_samples, dimension(),
                                               [](const Index, const Index) { return uniform(random::gen); })
                               .array()
                               .pow(1.0 / dimension())};
  return (u.cwiseProduct(r).array().colwise() / norm.transpose().array()).rowwise() + center_.array();
}
bool SphereSet::operator()(ConstVectorRef x) const {
  // Check if the vector is in the sphere set
  LUCID_CHECK_ARGUMENT_CMP(x.size(), ==, center_.size());
  return (x - center_).squaredNorm() <= radius_ * radius_ + std::numeric_limits<Scalar>::epsilon();
}
Matrix SphereSet::lattice(const VectorI& points_per_dim, const bool endpoint) const {
  // TODO(tend): Implement a more efficient lattice generation. This is generic, but not optimal.
  //  We could limit ourself to a 1/2**d square and then apply it symmetrically to the rest of the space.
  const RectSet rect_set{center_.array() - radius_, center_.array() + radius_};
  const Matrix lattice{rect_set.lattice(points_per_dim, endpoint)};
  std::vector<Index> mask_rows;
  mask_rows.reserve(lattice.rows());
  for (Index i = 0; i < lattice.rows(); ++i) {
    if (this->contains(lattice.row(i))) mask_rows.push_back(i);
  }
  return lattice(mask_rows, Eigen::placeholders::all);
}
void SphereSet::change_size(ConstVectorRef delta_size) {
  LUCID_TRACE_FMT("({})", LUCID_FORMAT_MATRIX(delta_size));
  LUCID_CHECK_ARGUMENT_EQ(delta_size.size(), dimension());
  LUCID_CHECK_ARGUMENT((delta_size.array() == delta_size(0)).all(), "delta_size", "must be uniform for all dimensions");

  const Scalar max_delta_size = delta_size(0);
  LUCID_CHECK_ARGUMENT_CMP(radius_ + max_delta_size / 2.0, >=, 0);

  radius_ += max_delta_size / 2.0;

  LUCID_TRACE_FMT("=> {}", *this);
}
Vector SphereSet::general_lower_bound() const { return center_.array() - radius_; }
Vector SphereSet::general_upper_bound() const { return center_.array() + radius_; }

std::unique_ptr<Set> SphereSet::to_rect_set() const {
  return std::make_unique<RectSet>(general_lower_bound(), general_upper_bound());
}

std::string SphereSet::to_string() const {
  return fmt::format("SphereSet( center( [{}] ) radius( {} ) )", center_, radius_);
}
bool SphereSet::operator==(const Set& other) const {
  if (Set::operator==(other)) return true;
  if (const auto other_rect = dynamic_cast<const SphereSet*>(&other)) return *this == *other_rect;
  return false;
}

bool SphereSet::operator==(const SphereSet& other) const {
  return dimension() == other.dimension() && center_ == other.center_ && radius_ == other.radius_;
}

std::unique_ptr<Set> SphereSet::increase_size_impl(ConstVectorRef size_increase) const {
  if (const double size_increase_0 = size_increase(0);
      std::ranges::all_of(size_increase, [size_increase_0](const double val) { return val == size_increase_0; })) {
    return std::make_unique<SphereSet>(center_, radius_ + size_increase_0 / 2.0);
  }
  return std::make_unique<EllipseSet>(center_, Vector::Constant(center().size(), radius_) + size_increase / 2.0);
}

std::ostream& operator<<(std::ostream& os, const SphereSet& set) { return os << set.to_string(); }

}  // namespace lucid
