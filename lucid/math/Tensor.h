/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Tensor class.
 */
#pragma once

#include <complex>
#include <numeric>
#include <span>
#include <vector>

#include "TensorView.h"
#include "lucid/lib/eigen.h"
#include "lucid/util/concept.h"

namespace lucid {

/**
 * Lightweight tensor class.
 * It uses a strided vector to support any number of dimensions.
 * @tparam T type of the elements in the tensor
 */
template <IsAnyOf<int, float, double, std::complex<double>> T>
class Tensor {
  template <IsAnyOf<int, float, double, std::complex<double>> TT>
  friend class Tensor;

 public:
  /**
   * Construct a new Tensor object.
   * The rank of the tensor is determined by the size of the dimensions vector.
   * @param dims shape of the tensor. Each element is the size of the corresponding dimension
   */
  explicit Tensor(std::vector<std::size_t> dims);
  /**
   * Construct a new Tensor object.
   * The rank of the tensor is determined by the size of the dimensions vector.
   * @param value value of all the elements in the tensor
   * @param dims shape of the tensor. Each element is the size of the corresponding dimension
   */
  Tensor(const T& value, std::vector<std::size_t> dims);
  /**
   * Construct a new Tensor object.
   * The rank of the tensor is determined by the size of the dimensions vector.
   * @param data data of the tensor. It will be copied
   * @param dims shape of the tensor. Each element is the size of the corresponding dimension
   */
  Tensor(std::vector<T> data, std::vector<std::size_t> dims);

  /**
   * Reshape the tensor to the new dimensions.
   * The data is not modified, only the view is changed.
   * @code
   * Tensor<int> t{std::vector<int>{1, 2, 3, 4}, std::vector<std::size_t>{2, 2}};
   * // 1  2
   * // 3  4
   * t.reshape({4});
   * // 1
   * // 2
   * // 3
   * // 4
   * @endcode
   * @pre The number of elements must remain the same
   * @param dims new dimensions of the tensor
   * @return reference to this object
   */
  Tensor& reshape(std::vector<std::size_t> dims);
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
    return view_(i, is...);
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
    return view_(i, is...);
  }

  /**
   * Get the element in the tensor using the indices in each dimension.
   * @tparam Container container of the indices
   * @tparam I index type
   * @param indices indices in each dimension
   * @return element in the tensor
   */
  template <template <class...> class Container, class I>
  const T& operator()(const Container<I>& indices) const {
    return view_(indices);
  }
  /**
   * Get the element in the tensor using the indices in each dimension.
   * @tparam Container container of the indices
   * @tparam I index type
   * @param indices indices in each dimension
   * @return element in the tensor
   */
  template <template <class...> class Container, std::convertible_to<const std::size_t> I>
  T& operator()(const Container<I>& indices) {
    return view_(indices);
  }

  /** @getter{size, tensor} */
  [[nodiscard]] std::size_t size() const { return data_.size(); }
  /** @getter{rank, tensor} */
  [[nodiscard]] std::size_t rank() const { return view_.rank(); }
  /** @getter{axes, tensor} */
  [[nodiscard]] const std::vector<std::size_t>& axes() const { return view_.axes(); }
  /** @getter{dimensions, tensor} */
  [[nodiscard]] const std::vector<std::size_t>& dimensions() const { return view_.dimensions(); }
  /** @getter{data, tensor} */
  [[nodiscard]] const std::vector<T>& data() const { return data_; }
  /** @getter{view, tensor} */
  [[nodiscard]] const TensorView<T>& view() const { return view_; }

  /**
   * Pad the tensor with a value.
   * The padding is applied to each dimension, and it is specified by a pair of indices,
   * one for the beginning and one for the end of the dimension.
   * @code
   * Tensor<int> t{std::vector<int>{1, 2, 3, 4}, std::vector<std::size_t>{2, 2}};
   * // 1  2
   * // 3  4
   * Tensor<int> padded{t.pad({{1, 2}, {0, 3}}, 0)};
   * // 0  0  0  0  0
   * // 1  2  0  0  0
   * // 3  4  0  0  0
   * // 0  0  0  0  0
   * // 0  0  0  0  0
   * @endcode
   * @param padding padding for each dimension
   * @param value value to fill the padding
   * @return padded tensor
   */
  [[nodiscard]] Tensor<T> pad(const std::vector<std::pair<Index, Index>>& padding, const T& value = {}) const;

  /**
   * Apply the Fast Fourier Transform to the tensor.
   * It is just the application of the FFT to each dimension of the tensor.
   * @code
   * Tensor<double> t{std::vector<double>{1, 2, 3, 4}, std::vector<std::size_t>{2, 2}};
   * // 1  2
   * // 3  4
   * Tensor<std::complex<double>> fft{t.fft()};
   * // (10 + 0i)  (-2 + 0i)
   * // (-4 + 0i)  (0 + 0i)
   * @endcode
   * @param axes axes to apply the FFT. Can be used to specify a different order of the dimensions
   * @return tensor with the FFT applied to each dimension
   * @see ifft
   */
  [[nodiscard]] Tensor<std::complex<double>> fft(const std::vector<std::size_t>& axes = {}) const;
  /**
   * Apply the Inverse Fast Fourier Transform to the tensor.
   * It is just the application of the IFFT to each dimension of the tensor.
   * @code
   * Tensor<double> t{std::vector<double>{1, 2, 3, 4}, std::vector<std::size_t>{2, 2}};
   * // 1  2
   * // 3  4
   * Tensor<std::complex<double>> fft{t.fft()};
   * // (10 + 0i)  (-2 + 0i)
   * // (-4 + 0i)  (0 + 0i)
   * Tensor<double> ifft{fft.ifft()};
   * // 1  2
   * // 3  4
   * @endcode
   * @param axes axes to apply the IFFT. Can be used to specify a different order of the dimensions
   * @return tensor with the IFFT applied to each dimension
   * @see fft
   */
  [[nodiscard]] Tensor<double> ifft(const std::vector<std::size_t>& axes = {}) const;

  operator Eigen::Map<const Eigen::VectorX<T>>() const { return view_; }
  operator Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>() const { return view_; }

 private:
  std::vector<T> data_;  ///< Data of the tensor
  TensorView<T> view_;   ///< View of the tensor
};

template <IsAnyOf<int, float, double, std::complex<double>> T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor);

extern template class Tensor<int>;
extern template class Tensor<float>;
extern template class Tensor<double>;
extern template class Tensor<std::complex<double>>;

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Tensor<int>)
OSTREAM_FORMATTER(lucid::Tensor<float>)
OSTREAM_FORMATTER(lucid::Tensor<double>)
OSTREAM_FORMATTER(lucid::Tensor<std::complex<double>>)

#endif
