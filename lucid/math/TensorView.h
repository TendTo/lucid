/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Tensor class.
 */
#pragma once

#include <complex>
#include <iosfwd>
#include <numeric>
#include <span>
#include <vector>

#include "lucid/lib/eigen.h"
#include "lucid/util/concept.h"
#include "lucid/util/exception.h"

namespace lucid {

/**
 * Lightweight wrapper that provides a view of a tensor.
 * It uses a strided vector to support any number of dimensions.
 * @note The data is not copied, so it must be kept alive and unchanged while the tensor is in use.
 * If you need an owning data structure use the Tensor class instead.
 * @tparam T type of the elements in the tensor
 */
template <IsAnyOf<int, float, double, std::complex<double>> T>
class TensorView {
  template <IsAnyOf<int, float, double, std::complex<double>> TT>
  friend class TensorView;

 public:
  /**
   * Construct a new Tensor object.
   * The rank of the tensor is determined by the size of the dimensions vector.
   * @param data data of the tensor
   * @param dims shape of the tensor. Each element is the size of the corresponding dimension
   */
  TensorView(std::span<const T> data, std::vector<std::size_t> dims);

  /**
   * Reshape the tensor to the new dimensions.
   * @pre The number of elements must remain the same
   * @param dims new dimensions of the tensor
   * @return reference to this object
   */
  TensorView& reshape(std::vector<std::size_t> dims);

  /**
   * Get the element in the tensor using the indices in each dimension.
   * @tparam I index type
   * @tparam Is variadic index types
   * @param i index in the first dimension
   * @param is indices in the remaining dimensions
   * @return element in the tensor
   */
  template <std::convertible_to<const std::size_t> I, class... Is>
  const T& operator()(I i, Is... is) const {
    return data_[index<0>(i, is...)];
  }
  /**
   * Get the element in the tensor using the indices in each dimension.
   * @tparam I index type
   * @tparam Is variadic index types
   * @param i index in the first dimension
   * @param is indices in the remaining dimensions
   * @return element in the tensor
   */
  template <std::convertible_to<const std::size_t> I, class... Is>
  T& operator()(I i, Is... is) {
    return const_cast<T&>(data_[index<0>(i, is...)]);
  }
  /**
   * Get the element in the tensor using the indices in each dimension.
   * @tparam I index type
   * @param indices indices in each dimension
   * @return element in the tensor
   */
  template <template <class...> class Container, std::convertible_to<const std::size_t> I>
  const T& operator()(const Container<I>& indices) const {
    return data_[index(std::span<const I>{indices})];
  }
  /**
   * Get the element in the tensor using the indices in each dimension.
   * @tparam I index type
   * @param indices indices in each dimension
   * @return element in the tensor
   */
  template <template <class...> class Container, std::convertible_to<const std::size_t> I>
  T& operator()(const Container<I>& indices) {
    return data_[index(std::span<const I>{indices})];
  }

  /**
   * Get the element in the underlying data vector using its index.
   * @note This is not the same as the tensor index, rather the linearized index
   * @tparam I index type
   * @param i index in the data vector
   * @return element in the tensor
   */
  template <std::convertible_to<const std::size_t> I>
  const T& operator[](I i) const {
    return data_[i];
  }

  /** @getter{size, tensor} */
  [[nodiscard]] std::size_t size() const { return data_.size(); }
  /** @getter{rank, tensor} */
  [[nodiscard]] std::size_t rank() const { return dims_.size(); }
  /** @getter{axes, tensor} */
  [[nodiscard]] const std::vector<std::size_t>& axes() const { return axes_; }
  /** @getter{dimensions, tensor} */
  [[nodiscard]] const std::vector<std::size_t>& dimensions() const { return dims_; }
  /** @getter{data, tensor} */
  [[nodiscard]] const std::span<const T>& data() const { return data_; }
  /**
   * Apply the Fast Fourier Transform to the tensor.
   * It is just the application of the FFT to each dimension of the tensor.
   * @pre The output tensor must have the same shape as the input tensor
   * @param[out] out tensor with the FFT applied to each dimension
   * @param coeff coefficient applied to the FFT. If NaN, the default coefficient, 1.0, is used
   * @see ifft
   */
  void fft(TensorView<std::complex<double>>& out, double coeff = std::numeric_limits<double>::quiet_NaN()) const;
  /**
   * Apply the Inverse Fast Fourier Transform to the tensor.
   * It is just the application of the IFFT to each dimension of the tensor.
   * @pre The output tensor must have the same shape as the input tensor
   * @param[out] out tensor with the IFFT applied to each dimension
   * @param coeff coefficient applied to the FFT. If NaN, the default coefficient, 1.0 / size(), is used
   * @see fft
   */
  void ifft(TensorView<double>& out, double coeff = std::numeric_limits<double>::quiet_NaN()) const;
  /**
   * Apply the Inverse Fast Fourier Transform to the tensor.
   * It is just the application of the IFFT to each dimension of the tensor.
   * @pre The output tensor must have the same shape as the input tensor
   * @param[out] out tensor with the IFFT applied to each dimension
   * @param coeff coefficient applied to the FFT. If NaN, the default coefficient, 1.0 / size(), is used
   * @see fft
   */
  void ifft(TensorView<std::complex<double>>& out, double coeff = std::numeric_limits<double>::quiet_NaN()) const;

  /**
   * Pad the tensor with a value.
   * The padding is applied to each dimension, and it is specified by a pair of indices,
   * one for the beginning and one for the end of the dimension.
   * @pre The `out` tensor must have the same dimensions as the input tensor plus `padding`
   * @param[out] out padded output tensor
   * @param padding padding for each dimension
   * @return padded tensor
   */
  void pad(TensorView<T>& out, const std::vector<std::pair<Index, Index>>& padding) const;
  /**
   * Pad the tensor with a value.
   * The `padding` is applied to each dimension starting at the respective `start_padding` index.
   * This allows the `padding` to be placed in the middle of the tensor.
   * @note Setting `start_padding` to 0 (the size of that dimension)
   * will place all the padding at the start (the end) of the dimension.
   * @pre The `out` tensor must have the same dimensions as the input tensor plus `padding`
   * @param[out] out padded output tensor
   * @param padding padding for each dimension
   * @param start_padding the index where the padding starts for each dimension
   * @return padded tensor
   */
  void pad(TensorView<T>& out, const std::vector<Index>& padding, const std::vector<Index>& start_padding) const;

  /**
   * Use the Fast Fourier Transform to upsample the tensor.
   * This procedure will interpolate the signal in the frequency domain and then apply the inverse FFT.
   * @pre The `out` tensor must have the same shape as the input tensor plus a non-negative `padding`
   * @warning If the functions making up the signal are not periodic or their frequency is too high
   * (> Nyquist frequency), the result will not be inaccurate.
   * @param[out[ out upsampled tensor
   * @see fft
   * @see ifft
   */
  void fft_upsample(TensorView<double>& out) const;

  operator Eigen::Map<const Eigen::VectorX<T>>() const { return {data_.data(), static_cast<Index>(data_.size())}; }
  operator Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>() const;

 private:
  /**
   * Get the linear index of an element in the data vector using the indices in each dimension.
   * @tparam Dim index of the current dimension
   * @tparam I index type
   * @tparam Is variadic index types
   * @param i index in the current dimension
   * @param is indices in the remaining dimensions
   * @return linear index of the element in the data vector
   */
  template <int Dim, std::convertible_to<const std::size_t> I, class... Is>
  [[nodiscard]] Index index(I i, Is... is) const {
#ifndef NCHECK
    if (i >= static_cast<I>(dims_[Dim])) throw exception::LucidInvalidArgumentException("Index out of bounds");
#endif
    if constexpr (sizeof...(is) == 0) {
      return strides_[Dim] * i;
    } else {
      return strides_[Dim] * i + index<Dim + 1>(is...);
    }
  }

  /**
   * Get the linear index of an element in the data vector using the indices in each dimension.
   * @tparam I index type
   * @param indices indices in each dimension
   * @return linear index of the element in the data vector
   */
  template <std::convertible_to<const std::size_t> I>
  [[nodiscard]] std::size_t index(std::span<const I> indices) const {
    if (indices.size() != dims_.size()) {
      throw exception::LucidInvalidArgumentException("Number of indices must match the number of dimensions");
    }
    std::size_t idx = 0;
    for (std::size_t i = 0; i < indices.size(); i++) {
#ifndef NCHECK
      if (indices[i] >= static_cast<I>(dims_[i])) throw exception::LucidInvalidArgumentException("Index out of bounds");
#endif
      idx += indices[i] * strides_[i];
    }
    return idx;
  }

  std::span<const T> data_;        ///< Data of the tensor
  std::vector<std::size_t> dims_;  ///< Shape of the tensor. Each element is the size of the corresponding dimension
  std::vector<std::size_t> axes_;  ///< Axes of the tensor. Goes from 0 to rank - 1
  std::vector<Index> strides_;     ///< Strides of the tensor. Used to calculate the index of an element
};

template <IsAnyOf<int, float, double, std::complex<double>> T>
std::ostream& operator<<(std::ostream& os, const TensorView<T>& tensor);

extern template class TensorView<int>;
extern template class TensorView<float>;
extern template class TensorView<double>;
extern template class TensorView<std::complex<double>>;

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::TensorView<int>)
OSTREAM_FORMATTER(lucid::TensorView<float>)
OSTREAM_FORMATTER(lucid::TensorView<double>)
OSTREAM_FORMATTER(lucid::TensorView<std::complex<double>>)

#endif
