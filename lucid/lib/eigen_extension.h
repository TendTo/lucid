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
/**
 * @namespace lucid::internal
 * Internal implementation details for Eigen extensions.
 */
namespace internal {

/**
 * Helper struct for circulant matrix type deduction.
 * @tparam ArgType type of the input vector
 */
template <class ArgType>
struct circulant_helper {
  using MatrixType = Eigen::Matrix<typename ArgType::Scalar, ArgType::SizeAtCompileTime, ArgType::SizeAtCompileTime,
                                   Eigen::ColMajor, ArgType::MaxSizeAtCompileTime, ArgType::MaxSizeAtCompileTime>;
};

/**
 * Functor for generating circulant matrices from a vector.
 * A circulant matrix is constructed by cyclically shifting the elements of a vector.
 * @tparam ArgType type of the input vector
 */
template <class ArgType>
class circulant_functor {
 public:
  /**
   * Construct a circulant functor.
   * @param arg input vector from which to generate the circulant matrix
   */
  explicit circulant_functor(const ArgType& arg) : arg_(arg) {}

  /**
   * Get the element at position (row, col) in the circulant matrix.
   * @param row row index
   * @param col column index
   * @return scalar value at the specified position
   */
  typename ArgType::Scalar operator()(const Eigen::Index row, const Eigen::Index col) const {
    Eigen::Index index = row - col;
    if (index < 0) index += arg_.size();
    return arg_(index);
  }

 private:
  const ArgType& arg_;  ///< Reference to the input vector
};

/**
 * Helper struct for shift matrix type deduction.
 * @tparam ArgType type of the input matrix
 */
template <class ArgType>
struct shift_helper {
  using MatrixType = Eigen::Matrix<typename ArgType::Scalar, ArgType::RowsAtCompileTime, ArgType::ColsAtCompileTime,
                                   Eigen::ColMajor, ArgType::MaxRowsAtCompileTime, ArgType::MaxColsAtCompileTime>;
};

/**
 * Functor for shifting matrix elements cyclically.
 * Elements are shifted by the specified number of rows and columns.
 * @tparam ArgType type of the input matrix
 */
template <class ArgType>
class shift_functor {
 public:
  /**
   * Construct a shift functor.
   * @param arg input matrix to shift
   * @param shift_rows number of rows to shift (positive = down, negative = up)
   * @param shift_cols number of columns to shift (positive = right, negative = left)
   */
  shift_functor(const ArgType& arg, const Eigen::Index shift_rows, const Eigen::Index shift_cols)
      : arg_{arg}, shift_rows_{shift_rows}, shift_cols_{shift_cols} {}

  /**
   * Get the element at position (row, col) in the shifted matrix.
   * @param row row index
   * @param col column index
   * @return scalar value at the specified position after shifting
   */
  typename ArgType::Scalar operator()(const Eigen::Index row, const Eigen::Index col) const {
    Eigen::Index shift_row = row + shift_rows_;
    Eigen::Index shift_col = col + shift_cols_;
    while (shift_row < 0) shift_row += arg_.rows();
    while (shift_col < 0) shift_col += arg_.cols();
    while (shift_row >= arg_.rows()) shift_row -= arg_.rows();
    while (shift_col >= arg_.cols()) shift_col -= arg_.cols();
    return arg_(shift_row, shift_col);
  }

 private:
  const ArgType& arg_;             ///< Reference to the input matrix
  const Eigen::Index shift_rows_;  ///< Number of rows to shift
  const Eigen::Index shift_cols_;  ///< Number of columns to shift
};

/**
 * Helper struct for pad matrix type deduction.
 * @tparam ArgType type of the input matrix
 */
template <class ArgType>
struct pad_helper {
  // TODO(tend): to optimise it further, we could introduce some compile time pad overloads and adjust the
  // new matrix size accordingly. Probably not really worth the effort for now.
  using MatrixType = Eigen::Matrix<typename ArgType::Scalar, Eigen::Dynamic, Eigen::Dynamic>;
};

/**
 * Functor for padding matrices with zeros.
 * Pads the input matrix by adding rows and columns of zeros around the borders.
 * @tparam ArgType type of the input matrix
 */
template <class ArgType>
class pad_functor {
 public:
  /**
   * Construct a pad functor.
   * @param arg input matrix to pad
   * @param pad_top number of zero rows to add at the top
   * @param pad_bottom number of zero rows to add at the bottom
   * @param pad_left number of zero columns to add at the left
   * @param pad_right number of zero columns to add at the right
   * @param value value to use for padding (typically 0)
   */
  pad_functor(const ArgType& arg, const Eigen::Index pad_top, const Eigen::Index pad_bottom,
              const Eigen::Index pad_left, const Eigen::Index pad_right, const typename ArgType::Scalar& value)
      : arg_{arg},
        pad_top_{pad_top},
        pad_bottom_{pad_bottom},
        pad_left_{pad_left},
        pad_right_{pad_right},
        value_{value} {}

  /**
   * Get the element at position (row, col) in the padded matrix.
   * @param row row index
   * @param col column index
   * @return scalar value at the specified position (either from original matrix or padding value)
   */
  typename ArgType::Scalar operator()(const Eigen::Index row, const Eigen::Index col) const {
    return row < pad_top_ || row >= arg_.rows() + pad_top_ || col < pad_left_ || col >= arg_.cols() + pad_left_
               ? value_
               : arg_(row - pad_top_, col - pad_left_);
  }

 private:
  const ArgType& arg_;                    ///< Reference to the input matrix
  const Eigen::Index pad_top_;            ///< Number of rows to pad at the top
  const Eigen::Index pad_bottom_;         ///< Number of rows to pad at the bottom
  const Eigen::Index pad_left_;           ///< Number of columns to pad at the left
  const Eigen::Index pad_right_;          ///< Number of columns to pad at the right
  const typename ArgType::Scalar value_;  ///< Value to use for padding
};

/** Type alias for pad operation. */
template <class ArgType>
using PadtOp = Eigen::CwiseNullaryOp<pad_functor<ArgType>, typename pad_helper<ArgType>::MatrixType>;

/** Type alias for shift operation. */
template <class ArgType>
using ShiftOp = Eigen::CwiseNullaryOp<shift_functor<ArgType>, typename shift_helper<ArgType>::MatrixType>;

/** Type alias for circulant operation. */
template <class ArgType>
using CirculantOp = Eigen::CwiseNullaryOp<circulant_functor<ArgType>, typename circulant_helper<ArgType>::MatrixType>;

}  // namespace internal

/**
 * Create a circulant matrix from a vector `arg`.
 * A circulant matrix is a square matrix where each row is a right cyclic shift of the previous row.
 * @tparam ArgType type of the vector
 * @param arg vector
 * @return circulant matrix
 */
template <class ArgType>
internal::CirculantOp<ArgType> circulant(const Eigen::MatrixBase<ArgType>& arg) {
  using MatrixType = typename internal::circulant_helper<ArgType>::MatrixType;
  return MatrixType::NullaryExpr(arg.size(), arg.size(), internal::circulant_functor<ArgType>{arg.derived()});
}

/**
 * Shift the rows and columns of a matrix or vector `x` by `shift_rows` and `shift_cols` respectively.
 * The sign of the shift determines the direction of the shift.
 * Overflows are wrapped around.
 * @tparam ArgType type of the matrix or vector
 * @param x matrix or vector
 * @param shift_rows shift rows. If negative, the rows are shifted up.
 * @param shift_cols shift columns. If negative, the columns are shifted left.
 * @return shifted matrix or vector
 */
template <class ArgType>
internal::ShiftOp<ArgType> shift(const Eigen::MatrixBase<ArgType>& x, const Eigen::Index shift_rows,
                                 const Eigen::Index shift_cols) {
  using MatrixType = typename internal::shift_helper<ArgType>::MatrixType;
  return MatrixType::NullaryExpr(x.rows(), x.cols(),
                                 internal::shift_functor<ArgType>{x.derived(), shift_rows, shift_cols});
}

/**
 * Rearrange a Fourier transform `x` by shifting the zero-frequency component to the center of the array.
 * It is useful for visualizing the Fourier transform with the zero-frequency component in the center.
 * Undoes the result of @ref ifftshift.
 * - If `x` is a vector, then `fftshift` swaps the left and right halves of `x`.
 * - If `x` is a matrix, then `fftshift` swaps the first quadrant of `x` with the third,
 * and the second quadrant with the fourth.
 *
 * @tparam ArgType type of the matrix or vector
 * @param x matrix or vector
 * @return shifted matrix or vector
 * @see https://www.mathworks.com/help/matlab/ref/fftshift.html
 */
template <class ArgType>
internal::ShiftOp<ArgType> fftshift(const Eigen::MatrixBase<ArgType>& x) {
  return shift(x, (x.rows() + 1) / 2, (x.cols() + 1) / 2);
}

/**
 * Rearranges a zero-frequency-shifted Fourier transform `x` back to the original transform output.
 * Undoes the result of @ref fftshift.
 * - If `x` is a vector, then `ifftshift` swaps the left and right halves of `x`.
 * - If `x` is a matrix, then `ifftshift` swaps the first quadrant of `x` with the third,
 * and the second quadrant with the fourth.
 *
 * @tparam ArgType type of the matrix or vector
 * @param x matrix or vector
 * @return shifted matrix or vector
 * @see https://www.mathworks.com/help/matlab/ref/ifftshift.html
 */
template <class ArgType>
internal::ShiftOp<ArgType> ifftshift(const Eigen::MatrixBase<ArgType>& x) {
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
internal::PadtOp<ArgType> pad(const Eigen::MatrixBase<ArgType>& x, const Eigen::Index pad_top,
                              const Eigen::Index pad_bottom, const Eigen::Index pad_left, const Eigen::Index pad_right,
                              const typename ArgType::Scalar& value) {
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
internal::PadtOp<ArgType> pad(const Eigen::MatrixBase<ArgType>& x, const Eigen::Index pad_size,
                              const typename ArgType::Scalar& value) {
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
internal::PadtOp<ArgType> pad(const Eigen::MatrixBase<ArgType>& x, const Eigen::Index pad_rows,
                              const Eigen::Index pad_cols, const typename ArgType::Scalar& value) {
  return pad(x, pad_rows, pad_rows, pad_cols, pad_cols, value);
}

}  // namespace lucid
