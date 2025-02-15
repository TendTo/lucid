/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/benchmark/InitBarr3Scenario.h"

#include "lucid/util/error.h"

namespace lucid::benchmark {
Matrix InitBarr3Scenario::operator()(ConstMatrixRef x) const { return f_stoch(x); }

Matrix InitBarr3Scenario::f_stoch(ConstMatrixRef x) {
  if (x.size() != 2) LUCID_INVALID_ARGUMENT_EXPECTED("x.size()", x.size(), 2);
  return f_det(x) + mvnrnd(Vector::Zero(2), Matrix::Identity(x.size(), x.size()) * 0.01);
}

Matrix InitBarr3Scenario::f_det(ConstMatrixRef x) {
  if (x.size() != 2) LUCID_INVALID_ARGUMENT_EXPECTED("x.size()", x.size(), 2);
  return Vector2{x.data()[1], -x.data()[0] - x.data()[1] + 1.0 / 3.0 * std::pow(x.data()[0], 3)};
}

}  // namespace lucid::benchmark
