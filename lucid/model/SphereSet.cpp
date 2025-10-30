/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * SphereSet class.
 */
#include "lucid/model/SphereSet.h"

#include <iostream>
#include <limits>
#include <vector>

#include "RectSet.h"
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
  [[maybe_unused]] const auto a = (x - center_).squaredNorm();
  return (x - center_).squaredNorm() <= radius_ * radius_ + std::numeric_limits<Scalar>::epsilon();
}
Matrix SphereSet::lattice(const VectorI& points_per_dim, const bool include_endpoints) const {
  // TODO(tend): Implement a more efficient lattice generation. This is generic, but not optimal.
  //  We could limit ourself to a 1/2**d square and then apply it symmetrically to the rest of the space.
  const RectSet rect_set{center_.array() - radius_, center_.array() + radius_};
  const Matrix lattice{rect_set.lattice(points_per_dim, include_endpoints)};
  std::vector<Index> mask_rows;
  mask_rows.reserve(lattice.rows());
  for (Index i = 0; i < lattice.rows(); ++i) {
    if (this->contains(lattice.row(i))) mask_rows.push_back(i);
  }
  return lattice(mask_rows, Eigen::all);
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

std::ostream& operator<<(std::ostream& os, const SphereSet& set) {
  return os << fmt::format("SphereSet( center( [{}] ) radius( {} ) )", set.center(), set.radius());
}

}  // namespace lucid
