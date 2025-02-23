/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Eigen wrapper.
 *
 * This header includes the eigen library and provides a various helpers.
 * Other files in the library should depend on this header instead of the GMP library directly.
 * Instead of including <eigen.h>, include "dlinear/libs/eigen.h".
 */
#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>

namespace lucid {
using Scalar = double;
using Matrix = Eigen::MatrixX<Scalar>;
using Vector = Eigen::VectorX<Scalar>;
using VectorBlock = Eigen::VectorBlock<Vector>;
using MatrixBlock = Eigen::Block<Matrix>;
using ConstVectorBlock = Eigen::VectorBlock<const Vector>;
using ConstMatrixBlock = Eigen::Block<const Matrix>;
using Vector2 = Eigen::Vector2d;
using Vector3 = Eigen::Vector3d;
using SMatrix = Eigen::SparseMatrix<Scalar>;
using SVector = Eigen::SparseVector<Scalar>;
using Index = Eigen::Index;
using Dimension = Eigen::Index;
using ConstMatrixRef = const Eigen::Ref<const Matrix>&;
using MatrixRef = Eigen::Ref<Matrix>;
using ConstVectorRef = const Eigen::Ref<const Vector>&;
using VectorRef = Eigen::Ref<Vector>;
template <class Derived>
using MatrixBase = Eigen::MatrixBase<Derived>;

/**
 * Create a 2D grid coordinates based on the coordinates contained in vectors `x` and `y`.
 * `X` is a matrix where each row is a copy of `x` and `Y` is a matrix where each column is a copy of `y`.
 * The grid represented by the coordinates `X` and `Y` has `y.size()` rows and `x.size()` columns.
 * @param x x coordinates
 * @param y y coordinates
 * @param[out] X x coordinates grid
 * @param[out] Y y coordinates grid
 * @see https://www.mathworks.com/help/matlab/ref/meshgrid.html
 */
inline void meshgrid(const Vector& x, const Vector& y, Matrix& X, Matrix& Y) {
  X = x.transpose().replicate(y.size(), 1);
  Y = y.replicate(1, x.size());
}
/**
 * Create a 2D grid coordinates based on the coordinates contained in vector `x`.
 * Equivalent to calling @ref meshgrid with as `meshgrid(x, x, X, Y)`.
 * @param x x coordinates
 * @param[out] X x coordinates grid
 * @param[out] Y y coordinates grid
 * @see https://www.mathworks.com/help/matlab/ref/meshgrid.html
 */
inline void meshgrid(const Vector& x, Matrix& X, Matrix& Y) { meshgrid(x, x, X, Y); }
/**
 * Create an evenly spaced values within a given interval.
 * @param low start of interval. The interval includes this value
 * @param high end of interval. The interval does not include this value, unless `with_last` is true
 * @param step spacing between values
 * @param with_last whether to include the last value in the interval
 * @return vector of evenly spaced values
 * @see https://numpy.org/doc/stable/reference/generated/numpy.arange.html
 */
inline Vector arange(const Scalar low, Scalar high, const Scalar step = 1, const bool with_last = false) {
  high -= (with_last) ? 0 : step;
  const Index N = static_cast<Index>(std::floor((high - low) / step) + 1);
  return Vector::LinSpaced(N, low, high);
}

/**
 * Peaks function defined over a pair of vectors.
 * Useful for demonstrating graphics functions, such as contour, mesh, pcolor, and surf.
 * It is obtained by translating and scaling Gaussian distributions and is defined as
 * @f[
 * f(x, y) = 3(1 - x)^2 \exp(-x^2 - (y + 1)^2) - 10\left(\frac{x}{5} - x^3 - y^5\right) \exp(-x^2 - y^2) - \frac{1}{3}
 * \exp(-(x + 1)^2 - y^2)
 * @f]
 * The function will be computed element-wise, producing a vector of the same size as the input vectors.
 * @param x vector of x values
 * @param y vector of y values
 * @return vector obtained by the application of the peaks function component-wise over the two input vectors
 * @see https://www.mathworks.com/help/matlab/ref/peaks.html
 */
Vector peaks(const Vector& x, const Vector& y);
/**
 * Peaks function defined over a pair of meshgrids.
 * Useful for demonstrating graphics functions, such as contour, mesh, pcolor, and surf.
 * It is obtained by translating and scaling Gaussian distributions and is defined as
 * @f[
 * f(x, y) = 3(1 - x)^2 \exp(-x^2 - (y + 1)^2) - 10\left(\frac{x}{5} - x^3 - y^5\right) \exp(-x^2 - y^2) - \frac{1}{3}
 * \exp(-(x + 1)^2 - y^2)
 * @f]
 * The function will be computed element-wise, producing a grid of the same size as the input grids.
 * @param x 2D grid of x values
 * @param y 2D grid of y values
 * @return 2D grid obtained by the application of the peaks function component-wise over the two input grids
 * @see https://www.mathworks.com/help/matlab/ref/peaks.html
 */
Matrix peaks(const Matrix& x, const Matrix& y);
/**
 * The multivariate normal distribution is a generalization of the univariate normal distribution to two or more
 * variables.
 * It has two parameters, a mean vector @f$ \mu @f$ and a covariance matrix @f$ \Sigma @f$,
 * that are analogous to the mean and variance parameters of a univariate normal distribution.
 * The diagonal elements of @f$ \Sigma @f$ contain the variances for each variable,
 * and the off-diagonal elements of @f$ \Sigma @f$ contain the covariances between variables.
 * The probability density function (pdf) of the d-dimensional multivariate normal distribution is
 * @f[
 *  f(x | \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x -
 * \mu)\right)
 * @f]
 * Although the multivariate normal cdf does not have a closed form, mvncdf can compute cdf values numerically.
 * @param mu mean vector
 * @param sigma covariance matrix
 * @param seed random seed. If negative, the seed is not set
 * @return random samples from the multivariate normal distribution
 * @see https://www.mathworks.com/help/stats/mvnrnd.html
 */
Matrix mvnrnd(const Vector& mu, const Matrix& sigma, int seed = -1);

/**
 * Calculates differences between adjacent elements of `X` `n` times.
 * By default, diff operates along the rows of `X`.
 * @param m matrix
 * @param n number of differences
 * @param rowwise whether to calculate differences along the rows or columns
 * @return matrix of differences
 * @see https://www.mathworks.com/help/matlab/ref/double.diff.html
 */
inline Matrix diff(ConstMatrixRef m, const int n = 1, const bool rowwise = true) {
  if (n < 1) return m;
  if (!rowwise) return diff(m.transpose(), n).transpose();
  if (m.rows() == 0) return Matrix::Zero(0, m.cols());
  return diff(m.bottomRows(m.rows() - 1) - m.topRows(m.rows() - 1), n - 1, rowwise);
}
/**
 * Given @f$ m @f$ inputs `matrices`, where matrix @f$ M_i @f$ has @f$ n_i @f$ columns, return a matrix
 * with @f$ \Prod_{i = 0}^m n_i @f$ column vectors,
 * where the columns consist of all combinations found by combining one column vector from each input matrix.
 * @param m1 first matrix
 * @param m2 second matrix
 * @param matrices remaining matrices
 * @return matrix with all combinations of column vectors
 * @see https://www.mathworks.com/help/deeplearning/ref/combvec.html
 */
template <class... Ms>
Matrix combvec(ConstMatrixRef m1, ConstMatrixRef m2, const Ms&... matrices) {
  Matrix res{m1.rows() + m2.rows(), m1.cols() * m2.cols()};
  res.topRows(m1.rows()) = m1.replicate(1, m2.cols());
  for (Index i = 0; i < m2.cols(); i++) {
    res.block(m1.rows(), i * m1.cols(), m2.rows(), m1.cols()) = m2.col(i).replicate(1, m1.cols());
  }
  if constexpr (sizeof...(Ms) == 0)
    return res;
  else
    return combvec(res, matrices...);
}
/**
 * Given a @mxn matrix `m` treat each row vector as a separate matrix
 * and return their combination using @ref combvec(m1, m2, matrices...).
 * @param m matrix
 * @return matrix with all combinations of column vectors
 * @see https://www.mathworks.com/help/deeplearning/ref/combvec.html
 */
Matrix combvec(ConstMatrixRef m);
/**
 * Calculate the root mean square of the elements of a vector.
 * @param x vector
 * @return root mean square
 */
Scalar rms(ConstMatrixRef x);
/**
 * Compute the `p`-norm distance between every pair of row vectors in the input.
 * If the input has shape @f$ N \times M @f$ the output vector will contain @f$ \frac{1}{2}N(N-1) @f$ elements.
 * @f[
 * \text{{input}} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_N \end{bmatrix}
 * @f]
 * @tparam P norm order
 * @param x input matrix
 * @return vector of distances
 */
template <Dimension P = 2>
Vector pdist(ConstMatrixRef x) {
  Vector distances{x.rows() * (x.rows() - 1) / 2};
  for (Index i = 0; i < x.rows(); i++) {
    for (Index j = 0; j < i; j++) {
      distances(i * (i - 1) / 2 + j) = (x.row(i) - x.row(j)).lpNorm<P>();
    }
  }
  return distances;
}
/**
 * Compute the Cumulative distribution function (CDF) of the normal distribution at oll point listed in @x.
 * @param x points at which to evaluate the CDF
 * @param sigma_f @f$ \sigma_f @f$ value used in the normal distribution (mean)
 * @param sigma_l @f$ \sigma_l @f$ value used in the normal distribution (standard deviation)
 * @return vector of CDF values at each point in @x
 */
Vector normal_cdf(ConstVectorRef x, Scalar sigma_f, Scalar sigma_l);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Matrix)
OSTREAM_FORMATTER(lucid::Vector)
OSTREAM_FORMATTER(lucid::ConstMatrixRef)
OSTREAM_FORMATTER(lucid::ConstVectorRef)
OSTREAM_FORMATTER(Eigen::Block<const lucid::Matrix>)
OSTREAM_FORMATTER(Eigen::Block<const lucid::Vector>)
OSTREAM_FORMATTER(lucid::SMatrix)
OSTREAM_FORMATTER(lucid::SVector)

#endif
