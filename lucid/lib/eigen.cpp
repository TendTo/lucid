/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/lib/eigen.h"

#include <Eigen/Eigenvalues>
#include <random>
#include <utility>

#include "lucid/util/error.h"
#include "lucid/util/math.h"

namespace lucid {

namespace {
/**
 * @see https://stackoverflow.com/a/40245513/15153171
 */
struct normal_random_variable {
  explicit normal_random_variable(const Matrix& covar, const int seed = -1)
      : normal_random_variable(Vector::Zero(covar.rows()), covar, seed) {}

  normal_random_variable(Vector mean, const Matrix& covar, const int seed = -1)
      : mean_{std::move(mean)}, transform_{}, gen_{std::random_device{}()}, dist_{} {
    if (seed >= 0) gen_.seed(seed);
    const Eigen::SelfAdjointEigenSolver<Matrix> eigenSolver{covar};
    transform_ = (eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal()).matrix();
  }

  Vector mean_;
  Matrix transform_;
  mutable std::mt19937 gen_;
  mutable std::normal_distribution<> dist_;

  Vector operator()() const {
    return mean_ + transform_ * Vector{mean_.size()}.unaryExpr([&](auto) { return dist_(gen_); });
  }
};
}  // namespace

Vector peaks(const Vector& x, const Vector& y) {
  if (x.size() != y.size()) LUCID_INVALID_ARGUMENT_EXPECTED("x.size()", x.size(), y.size());
  Vector z{x.size()};
  for (Index i = 0; i < x.size(); i++) z(i) = peaks(x(i), y(i));
  return z;
}

Matrix peaks(const Matrix& x, const Matrix& y) {
  if (x.rows() != y.rows() || x.cols() != y.cols()) LUCID_INVALID_ARGUMENT("x and y", "they must have the same size");
  Matrix z{x.rows(), x.cols()};
  for (Index i = 0; i < x.size(); i++) z.data()[i] = peaks(x.data()[i], y.data()[i]);
  return z;
}

Matrix mvnrnd(const Vector& mu, const Matrix& sigma, const int seed) {
  return normal_random_variable{mu, sigma, seed}();
}
Scalar rms(ConstMatrixRef x) {
  Scalar sum = 0;
  for (Index i = 0; i < x.size(); i++) sum += x.data()[i] * x.data()[i];
  return std::sqrt(sum / static_cast<double>(x.size()));
}

}  // namespace lucid
