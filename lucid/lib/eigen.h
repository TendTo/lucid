/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Eigen wrapper.
 * This header includes the eigen library and provides a various helpers.
 * Other files in the library should depend on this header instead of the eigen library directly.
 * Instead of including <eigen.h>, include "lucid/lib/eigen.h".
 */
#pragma once

#include <fstream>
#include <ranges>
#include <span>
#include <string>

/**
 * Define the Eigen MatrixBase plugin header file path.
 * This plugin extends Eigen matrices with additional functionality like read/write methods.
 * Must be defined before including any Eigen headers.
 */
#define EIGEN_MATRIXBASE_PLUGIN "lucid/lib/eigen_matrix_base_plugin.h"

#include <Eigen/Cholesky>                  // NOLINT(build/include_order): must be after the plugin
#include <Eigen/Core>                      // NOLINT(build/include_order): must be after the plugin
#include <Eigen/LU>                        // NOLINT(build/include_order): must be after the plugin
#include <Eigen/SparseCore>                // NOLINT(build/include_order): must be after the plugin
#include <unsupported/Eigen/CXX11/Tensor>  // NOLINT(build/include_order): must be after the plugin
#include <unsupported/Eigen/FFT>           // NOLINT(build/include_order): must be after the plugin

#include "lucid/lib/eigen_extension.h"

namespace lucid {
using Scalar = double;
template <class T>
using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <class T>
using VectorT = Eigen::RowVectorX<T>;
using Matrix = MatrixT<Scalar>;
using Vector = VectorT<Scalar>;
using MatrixC = MatrixT<std::complex<Scalar>>;
using VectorC = VectorT<std::complex<Scalar>>;
using VectorI = VectorT<Eigen::Index>;
using VectorBlock = Eigen::VectorBlock<Vector>;
using MatrixBlock = Eigen::Block<Matrix>;
using ConstVectorBlock = Eigen::VectorBlock<const Vector>;
using ConstMatrixBlock = Eigen::Block<const Matrix>;
using Vector2 = Eigen::RowVector2d;
using Vector3 = Eigen::RowVector3d;
using Index = Eigen::Index;
using Dimension = Eigen::Index;
using ConstMatrixRef = const Eigen::Ref<const Matrix>&;
using ConstMatrixRefCopy = Eigen::Ref<const Matrix>;
using MatrixRef = Eigen::Ref<Matrix>;
using ConstVectorRef = const Eigen::Ref<const Vector>&;
using VectorRef = Eigen::Ref<Vector>;
using ConstMatrixRowIndexedView =
    decltype(std::declval<const Matrix>()(std::declval<std::vector<Index>>(), Eigen::indexing::all));
using ConstMatrixColIndexedView =
    decltype(std::declval<const Matrix>()(Eigen::indexing::all, std::declval<std::vector<Index>>()));
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
 * @note The seed for the random number generator can be set using @ref random::seed.
 * @param mu mean vector
 * @param sigma covariance matrix
 * @return random samples from the multivariate normal distribution
 * @see https://www.mathworks.com/help/stats/mvnrnd.html
 */
Matrix mvnrnd(const Vector& mu, const Matrix& sigma);

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
 * with @f$ \Pi_{i = 0}^m n_i @f$ column vectors,
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
 * Given a matrix `m` treat each row vector as a separate matrix
 * and return their combination using @ref combvec.
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
 * Compute the median of a vector or matrix.
 * If the vector has an even number of elements, the median is the mean of the two middle elements.
 * @warning The storage order of the matrix matters, while it changes nothing for vectors.
 * @param x vector or matrix
 * @return median of the vector or matrix
 * @see https://stackoverflow.com/a/62698308/15153171
 */
inline Scalar median(Matrix x) {
  std::span<Scalar> distances{x.data(), static_cast<std::size_t>(x.size())};

  const auto middle_it = distances.begin() + distances.size() / 2;
  std::ranges::nth_element(distances, middle_it);

  if (distances.size() & 1) return *middle_it;                                // odd => middle element
  return (*std::max_element(distances.begin(), middle_it) + *middle_it) / 2;  // even => mean
}
/**
 * Compute the `p`-norm distance between every pair of row vectors in the input.
 * If the input has shape @f$ N \times M @f$ the output vector will contain @f$ \frac{1}{2}N(N-1) @f$ elements
 * or be a lower triangular matrix, depending on `TriangularMatrix`.
 * @f[
 * \text{{input}} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_N \end{bmatrix}
 * @f]
 * @todo Remove TriangularMatrix tparam in favor of `squareform` function
 * @tparam P norm order
 * @tparam Squared whether to compute the squared distance. Only available for `p` == 2
 * @tparam TriangularMatrix whetehr to return a lower triangular matrix with all the distances
 * or a vector view of the strictly lower section of the full matrix
 * @param x input matrix
 * @return vector of distances or lower triangular matrix.
 */
template <Dimension P = 2, bool Squared = false, bool TriangularMatrix = false, class Derived>
  requires(P > 0) && (!Squared || P == 2)
auto pdist(const Eigen::MatrixBase<Derived>& x) {
  if constexpr (TriangularMatrix) {
    Matrix distances{x.rows(), x.rows()};
    distances.diagonal() = Vector::Zero(x.rows());
    for (Index i = 0; i < x.rows(); i++) {
      for (Index j = 0; j < i; j++) {
        if constexpr (Squared) {
          distances(i, j) = (x.row(i) - x.row(j)).squaredNorm();
        } else {
          distances(i, j) = (x.row(i) - x.row(j)).template lpNorm<P>();
        }
      }
    }
    distances.triangularView<Eigen::StrictlyUpper>() = distances.triangularView<Eigen::StrictlyLower>().transpose();
    return distances;
  } else {
    Vector distances{x.rows() * (x.rows() - 1) / 2};
    for (Index i = 0; i < x.rows(); i++) {
      for (Index j = 0; j < i; j++) {
        if constexpr (Squared) {
          distances(i * (i - 1) / 2 + j) = (x.row(i) - x.row(j)).squaredNorm();
        } else {
          distances(i * (i - 1) / 2 + j) = (x.row(i) - x.row(j)).template lpNorm<P>();
        }
      }
    }
    return distances;
  }
}
/**
 * Compute the `p`-norm distance between every pair of row vectors in the input matrices.
 * @tparam DerivedX type of the first input matrix
 * @tparam DerivedY type of the second input matrix
 * @tparam P norm order
 * @tparam Squared whether to compute the squared distance. Only available for `p` == 2
 * @param x input matrix
 * @param y input matrix
 * @return matrix of distances
 */
template <Dimension P = 2, bool Squared = false, class DerivedX, class DerivedY>
  requires(P > 0) && (!Squared || P == 2)
Matrix pdist(const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y) {
  Matrix distances{x.rows(), y.rows()};
  for (Index i = 0; i < x.rows(); i++) {
    for (Index j = 0; j < y.rows(); j++) {
      if constexpr (Squared) {
        distances(i, j) = (x.row(i) - y.row(j)).squaredNorm();
      } else {
        distances(i, j) = (x.row(i) - y.row(j)).template lpNorm<P>();
      }
    }
  }
  return distances;
}

/**
 * Compute the Cumulative distribution function (CDF) of the normal distribution at oll point listed in @x.
 * @param x points at which to evaluate the CDF
 * @param sigma_f @sigmaf value used in the normal distribution (mean)
 * @param sigma_l @sigmal value used in the normal distribution (standard deviation)
 * @return vector of CDF values at each point in @x
 */
Vector normal_cdf(ConstVectorRef x, Scalar sigma_f, Scalar sigma_l);

/**
 * Perform a 2D Fast Fourier Transform (FFT) on the input matrix.
 * @param x input matrix
 * @return FFT of the input matrix
 */
inline Eigen::MatrixXcd fft2(const Matrix& x) {
  Eigen::RowVectorXcd tempRow;
  Eigen::RowVectorXcd tempCol;
  Eigen::MatrixXcd f_fft{x.rows(), x.cols()};

  Eigen::FFT<double> fft;
  for (int k = 0; k < x.rows(); k++) {
    fft.fwd(tempRow, x.row(k).eval());
    f_fft.row(k) = tempRow;
  }
  for (int k = 0; k < x.cols(); k++) {
    fft.fwd(tempCol, f_fft.col(k));
    f_fft.col(k) = tempCol;
  }
  return f_fft;
}

/**
 * Perform a 2D inverse Fast Fourier Transform (FFT) on the input matrix.
 * @param x input matrix
 * @return inverse FFT of the input matrix
 */
inline Matrix ifft2(const Eigen::MatrixXcd& x) {
  Eigen::RowVectorXcd tempRow;
  Eigen::RowVectorXcd tempCol;
  Eigen::MatrixXcd i_fft{x.rows(), x.cols()};

  Eigen::FFT<double> fft;
  for (int k = 0; k < x.rows(); k++) {
    fft.inv(tempRow, x.row(k).eval());
    i_fft.row(k) = tempRow;
  }
  for (int k = 0; k < x.cols(); k++) {
    fft.inv(tempCol, i_fft.col(k));
    i_fft.col(k) = tempCol;
  }
  return i_fft.real();
}

template <typename T>
using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename Scalar, int rank, typename sizeType>
auto MatrixCast(const Eigen::Tensor<Scalar, rank>& tensor, const sizeType rows, const sizeType cols) {
  return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
}

template <typename Scalar>
auto TensorCast(const MatrixType<Scalar>& matrix) {
  return Eigen::TensorMap<Eigen::Tensor<const Scalar, 2>>(matrix.data(), std::array{matrix.rows(), matrix.cols()});
}

template <typename Scalar>
auto TensorCast(const std::span<Scalar>& data, std::array<Index, 2> shape) {
  return Eigen::TensorMap<Eigen::Tensor<const Scalar, 2>>(data.data(), shape);
}

inline Eigen::MatrixXcd fftn(const Matrix& x) {
  Eigen::Tensor<double, 2> t{TensorCast(x)};
  Eigen::Tensor<std::complex<double>, 2> res = t.fft<Eigen::BothParts, Eigen::FFT_FORWARD>(std::array{0, 1});
  return MatrixCast(res, x.rows(), x.cols());
}

inline Eigen::MatrixXcd fftn(const std::span<const double>& x, const std::array<Index, 2> shape) {
  Eigen::Tensor<double, 2> t{TensorCast(x, shape)};
  Eigen::Tensor<std::complex<double>, 2> res = t.fft<Eigen::BothParts, Eigen::FFT_FORWARD>(std::array{0, 1});
  return MatrixCast(res, shape[0], shape[1]);
}

inline Matrix ifftn(const std::span<const std::complex<double>>& x, const std::array<Index, 2> shape) {
  Eigen::Tensor<std::complex<double>, 2> t{TensorCast(x, shape)};
  Eigen::Tensor<std::complex<double>, 2> res = t.fft<Eigen::BothParts, Eigen::FFT_REVERSE>(std::array{0, 1});
  return MatrixCast(res, shape[0], shape[1]).real();
}

inline Matrix ifftn(const MatrixC& x) {
  Eigen::Tensor<std::complex<double>, 2> t{TensorCast(x)};
  Eigen::Tensor<std::complex<double>, 2> res = t.fft<Eigen::BothParts, Eigen::FFT_REVERSE>(std::array{0, 1});
  return MatrixCast(res, x.rows(), x.cols()).real();
}

/**
 * Read a matrix from a file.
 * The file must have the following format:
 * ```txt
 * rowsXcols
 * element_1,element_2,...,element_n
 * ```
 * @tparam Scalar type of the matrix
 * @param file_name name of the file to read
 * @return matrix read from the file
 */
template <class Scalar>
Eigen::MatrixX<Scalar> read_matrix(const std::string_view file_name) {
  std::ifstream in_file(file_name.data());
  if (!in_file.is_open()) return Eigen::MatrixX<Scalar>{};
  std::string line;
  std::getline(in_file, line);
  std::istringstream iss{line};
  Index rows, cols;
  char x;
  iss >> rows >> x >> cols;
  Eigen::MatrixX<Scalar> m{rows, cols};
  std::getline(in_file, line);
  iss = std::istringstream{line};
  for (Index j = 0; j < cols; ++j) {
    for (Index i = 0; i < rows; ++i) {
      iss >> m(i, j) >> x;
    }
  }
  return m;
}

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Matrix)
OSTREAM_FORMATTER(lucid::Vector)
OSTREAM_FORMATTER(decltype(std::declval<lucid::Vector>().transpose()))
OSTREAM_FORMATTER(decltype(std::declval<lucid::Matrix>().transpose()))
OSTREAM_FORMATTER(lucid::ConstMatrixRef)
OSTREAM_FORMATTER(lucid::ConstVectorRef)
OSTREAM_FORMATTER(Eigen::Block<const lucid::Matrix>)
OSTREAM_FORMATTER(Eigen::Block<const lucid::Vector>)
OSTREAM_FORMATTER(Eigen::Ref<const lucid::Matrix>)
OSTREAM_FORMATTER(Eigen::Ref<const lucid::Vector>)

#endif
