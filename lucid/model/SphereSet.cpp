/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * SphereSet class.
 */
#include "lucid/model/SphereSet.h"

#include <iostream>

#include "lucid/util/error.h"
#include "lucid/util/random.h"

namespace lucid {

namespace {
std::uniform_real_distribution<> dis(0.0, 1.0);
}

SphereSet::SphereSet(ConstVectorRef center, Scalar radius) : center_(center), radius_(radius) {
  LUCID_CHECK_ARGUMENT_CMP(center.size(), >, 0);
  LUCID_CHECK_ARGUMENT_CMP(radius, >=, 0);
}
Matrix SphereSet::sample(const Index num_samples) const {
  Matrix u{Matrix::NullaryExpr(num_samples, dimension(), [](const Index, const Index) { return dis(random::gen); })};
  std::cout << "u: " << u << std::endl;
  const Vector norm{u.rowwise().norm()};
  std::cout << "norm: " << norm << std::endl;
  const Matrix r{
      radius_ * Matrix::NullaryExpr(num_samples, dimension(), [](const Index, const Index) { return dis(random::gen); })
                    .array()
                    .pow(1.0 / dimension())};
  std::cout << "r: " << r << std::endl;
  return u.cwiseProduct(r).cwiseQuotient(norm) + center_.replicate(num_samples, 1);
}
bool SphereSet::operator()(ConstVectorRef x) const {
  // Check if the vector is in the sphere set
  LUCID_CHECK_ARGUMENT_CMP(x.size(), ==, center_.size());
  return (x - center_).squaredNorm() <= radius_ * radius_;
}
Matrix SphereSet::lattice(const VectorI&, bool) const { LUCID_NOT_IMPLEMENTED(); }

}  // namespace lucid
