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
template <IsAnyOf<double, std::complex<double>> T>
class Tensor {
  template <IsAnyOf<double, std::complex<double>> TT>
  friend class Tensor;

 public:
  /**
   * Construct a new Tensor object.
   * The rank of the tensor is determined by the size of the dimensions vector.
   * @param dims shape of the tensor. Each element is the size of the corresponding dimension
   */
  explicit Tensor(std::vector<std::size_t> dims)
      : Tensor(std::vector<T>(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies{})), dims) {}
  /**
   * Construct a new Tensor object.
   * The rank of the tensor is determined by the size of the dimensions vector.
   * @param data data of the tensor
   * @param dims shape of the tensor. Each element is the size of the corresponding dimension
   */
  Tensor(std::vector<T> data, std::vector<std::size_t> dims);

  /**
   * Reshape the tensor to the new dimensions.
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
  template <class I, class... Is>
  const T& operator()(I i, Is... is) const {
    return view_(i, is...);
  }

  /**
   * Get the element in the tensor using the indices in each dimension.
   * @tparam I index type
   * @param indices indices in each dimension
   * @return element in the tensor
   */
  template <class I>
  const T& operator()(const std::vector<I>& indices) const {
    return operator()(std::span{indices});
  }
  /**
   * Get the element in the tensor using the indices in each dimension.
   * @tparam I index type
   * @param indices indices in each dimension
   * @return element in the tensor
   */
  template <class I>
  const T& operator()(const std::span<const I> indices) const {
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
   * Apply the Fast Fourier Transform to the tensor.
   * It is just the application of the FFT to each dimension of the tensor.
   * @return tensor with the FFT applied to each dimension
   * @see ifft
   */
  [[nodiscard]] Tensor<std::complex<double>> fft() const;
  /**
   * Apply the Inverse Fast Fourier Transform to the tensor.
   * It is just the application of the IFFT to each dimension of the tensor.
   * @return tensor with the IFFT applied to each dimension
   * @see fft
   */
  [[nodiscard]] Tensor<double> ifft() const;

  operator Eigen::Map<const Eigen::VectorX<T>>() const { return view_; }
  operator Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>() const { return view_; }

 private:
  std::vector<T> data_;  ///< Data of the tensor
  TensorView<T> view_;   ///< View of the tensor
};

extern template class Tensor<double>;
extern template class Tensor<std::complex<double>>;

}  // namespace lucid
