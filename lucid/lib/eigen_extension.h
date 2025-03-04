/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Eigen extensions and expressions.
 */

#pragma once
#include <Eigen/Core>

namespace lucid {
namespace internal {

template <class ArgType>
struct circulant_helper {
  using MatrixType = Eigen::Matrix<typename ArgType::Scalar, ArgType::SizeAtCompileTime, ArgType::SizeAtCompileTime,
                                   Eigen::ColMajor, ArgType::MaxSizeAtCompileTime, ArgType::MaxSizeAtCompileTime>;
};
template <class ArgType>
class circulant_functor {
 public:
  explicit circulant_functor(const ArgType& arg) : arg_(arg) {}
  typename ArgType::Scalar operator()(const Eigen::Index row, const Eigen::Index col) const {
    Eigen::Index index = row - col;
    if (index < 0) index += arg_.size();
    return arg_(index);
  }

 private:
  const ArgType& arg_;
};

template <class ArgType>
struct shift_helper {
  using MatrixType = Eigen::Matrix<typename ArgType::Scalar, ArgType::RowsAtCompileTime, ArgType::ColsAtCompileTime,
                                   Eigen::ColMajor, ArgType::MaxRowsAtCompileTime, ArgType::MaxColsAtCompileTime>;
};
template <class ArgType>
class shift_functor {
 public:
  shift_functor(const ArgType& arg, const Eigen::Index shift_rows, const Eigen::Index shift_cols)
      : arg_{arg}, shift_rows_{shift_rows}, shift_cols_{shift_cols} {}

  typename ArgType::Scalar operator()(const Eigen::Index row, const Eigen::Index col) const {
    Eigen::Index shift_row = row + shift_rows_;
    Eigen::Index shift_col = col + shift_cols_;
    if (shift_row < 0) shift_row += arg_.rows();
    if (shift_col < 0) shift_col += arg_.cols();
    if (shift_row >= arg_.rows()) shift_row -= arg_.rows();
    if (shift_col >= arg_.cols()) shift_col -= arg_.cols();
    return arg_(shift_row, shift_col);
  }

 private:
  const ArgType& arg_;
  const Eigen::Index shift_rows_;
  const Eigen::Index shift_cols_;
};

template <class ArgType>
struct pad_helper {
  // TODO(tend): to optimise it further, we could introduce some compile time pad overloads and adjust the
  // new matrix size accordingly. Probably not really worth the effort for now.
  using MatrixType = Eigen::Matrix<typename ArgType::Scalar, Eigen::Dynamic, Eigen::Dynamic>;
};
template <class ArgType>
class pad_functor {
 public:
  pad_functor(const ArgType& arg, const Eigen::Index pad_top, const Eigen::Index pad_bottom,
              const Eigen::Index pad_left, const Eigen::Index pad_right, const typename ArgType::Scalar& value)
      : arg_{arg},
        pad_top_{pad_top},
        pad_bottom_{pad_bottom},
        pad_left_{pad_left},
        pad_right_{pad_right},
        value_{value} {}

  typename ArgType::Scalar operator()(const Eigen::Index row, const Eigen::Index col) const {
    return row < pad_top_ || row >= arg_.rows() + pad_top_ || col < pad_left_ || col >= arg_.cols() + pad_left_
               ? value_
               : arg_(row - pad_top_, col - pad_left_);
  }

 private:
  const ArgType& arg_;
  const Eigen::Index pad_top_;
  const Eigen::Index pad_bottom_;
  const Eigen::Index pad_left_;
  const Eigen::Index pad_right_;
  const typename ArgType::Scalar value_;
};

}  // namespace internal

/**
 * Create a circulant matrix from a vector `arg`.
 * A circulant matrix is a square matrix where each row is a right cyclic shift of the previous row.
 * @tparam ArgType type of the vector
 * @param arg vector
 * @return circulant matrix
 */
template <class ArgType>
Eigen::CwiseNullaryOp<internal::circulant_functor<ArgType>, typename internal::circulant_helper<ArgType>::MatrixType>
circulant(const Eigen::MatrixBase<ArgType>& arg) {
  using MatrixType = typename internal::circulant_helper<ArgType>::MatrixType;
  return MatrixType::NullaryExpr(arg.size(), arg.size(), internal::circulant_functor<ArgType>{arg.derived()});
}

/**
 * Shift the rows and columns of a matrix or vector `x` by `shift_rows` and `shift_cols` respectively.
 * The sign of the shift determines the direction of the shift.
 * Overflows are wrapped around, as long as the shift is at most the size of the matrix or vector in that direction.
 * @tparam ArgType type of the matrix or vector
 * @param x matrix or vector
 * @param shift_rows shift rows. If negative, the rows are shifted up.
 * Must be in the range [-x.rows(), x.rows()).
 * @param shift_cols shift columns. If negative, the columns are shifted left.
 * Must be in the range [-x.cols(), x.cols()).
 * @return shifted matrix or vector
 */
template <class ArgType>
Eigen::CwiseNullaryOp<internal::shift_functor<ArgType>, typename internal::shift_helper<ArgType>::MatrixType> shift(
    const Eigen::MatrixBase<ArgType>& x, const Eigen::Index shift_rows, const Eigen::Index shift_cols) {
  using MatrixType = typename internal::shift_helper<ArgType>::MatrixType;
  return MatrixType::NullaryExpr(x.rows(), x.cols(),
                                 internal::shift_functor<ArgType>{x.derived(), shift_rows, shift_cols});
}

/**
 * Rearrange a Fourier transform `x` by shifting the zero-frequency component to the center of the array.
 * It is useful for visualizing the Fourier transform with the zero-frequency component in the center.
 * Undoes the result of @ref ifftshift.
 * - If `x` is a vector, then `fftshift` swaps the left and right halves of `x`.
 * 0 If `x` is a matrix, then `fftshift` swaps the first quadrant of `x` with the third,
 * and the second quadrant with the fourth.
 * @tparam ArgType type of the matrix or vector
 * @param x matrix or vector
 * @return shifted matrix or vector
 * @see https://www.mathworks.com/help/matlab/ref/fftshift.html
 */
template <class ArgType>
Eigen::CwiseNullaryOp<internal::shift_functor<ArgType>, typename internal::shift_helper<ArgType>::MatrixType> fftshift(
    const Eigen::MatrixBase<ArgType>& x) {
  return shift(x, (x.rows() + 1) / 2, (x.cols() + 1) / 2);
}

/**
 * Rearranges a zero-frequency-shifted Fourier transform `x` back to the original transform output.
 * Undoes the result of @ref fftshift.
 * - If `x` is a vector, then `ifftshift` swaps the left and right halves of `x`.
 * - If `x` is a matrix, then `ifftshift` swaps the first quadrant of `x` with the third,
 * and the second quadrant with the fourth.
 * @tparam ArgType type of the matrix or vector
 * @param x matrix or vector
 * @return shifted matrix or vector
 * @see https://www.mathworks.com/help/matlab/ref/ifftshift.html
 */
template <class ArgType>
Eigen::CwiseNullaryOp<internal::shift_functor<ArgType>, typename internal::shift_helper<ArgType>::MatrixType> ifftshift(
    const Eigen::MatrixBase<ArgType>& x) {
  return shift(x, x.rows() / 2, x.cols() / 2);
}

/**
 * Pad a matrix or vector `x` with `pad_top`, `pad_bottom`, `pad_left`, and `pad_right` rows and columns respectively.
 * The padding value is set to `value`.
 * @tparam ArgType type of the matrix or vector
 * @param x matrix or vector
 * @param pad_top padding rows at the top
 * @param pad_bottom padding rows at the bottom
 * @param pad_left padding columns at the left
 * @param pad_right padding columns at the right
 * @param value padding value
 * @return padded matrix or vector
 */
template <class ArgType>
Eigen::CwiseNullaryOp<internal::pad_functor<ArgType>, typename internal::pad_helper<ArgType>::MatrixType> pad(
    const Eigen::MatrixBase<ArgType>& x, const Eigen::Index pad_top, const Eigen::Index pad_bottom,
    const Eigen::Index pad_left, const Eigen::Index pad_right, const typename ArgType::Scalar& value) {
  using MatrixType = typename internal::pad_helper<ArgType>::MatrixType;
  return MatrixType::NullaryExpr(
      x.rows() + pad_top + pad_bottom, x.cols() + pad_left + pad_right,
      internal::pad_functor<ArgType>{x.derived(), pad_top, pad_bottom, pad_left, pad_right, value});
}

/**
 * Pad a matrix or vector `x` with `pad_size` rows and columns in all directions,
 * effectively adding `2 * pad_size` rows and columns to the input.
 * The padding value is set to `value`.
 * @tparam ArgType type of the matrix or vector
 * @param x matrix or vector
 * @param pad_size padding rows and columns to be added in all directions,
 * @param value padding value
 * @return padded matrix or vector
 */
template <class ArgType>
Eigen::CwiseNullaryOp<internal::pad_functor<ArgType>, typename internal::pad_helper<ArgType>::MatrixType> pad(
    const Eigen::MatrixBase<ArgType>& x, const Eigen::Index pad_size, const typename ArgType::Scalar& value) {
  return pad(x, pad_size, pad_size, pad_size, pad_size, value);
}

/**
 * Pad a matrix or vector `x` with `pad_rows` rows and `pad_cols` columns in both directions,
 * effectively adding `2 * pad_rows` rows and `2 * pad_cols` columns to the input.
 * The padding value is set to `value`.
 * @tparam ArgType type of the matrix or vector
 * @param x matrix or vector
 * @param pad_rows padding rows to be added in both directions
 * @param pad_cols padding columns to be added in both directions
 * @param value padding value
 * @return padded matrix or vector
 */
template <class ArgType>
Eigen::CwiseNullaryOp<internal::pad_functor<ArgType>, typename internal::pad_helper<ArgType>::MatrixType> pad(
    const Eigen::MatrixBase<ArgType>& x, const Eigen::Index pad_rows, const Eigen::Index pad_cols,
    const typename ArgType::Scalar& value) {
  return pad(x, pad_rows, pad_rows, pad_cols, pad_cols, value);
}

}  // namespace lucid
