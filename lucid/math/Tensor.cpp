/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/Tensor.h"

namespace lucid {

template <IsAnyOf<double, std::complex<double>> T>
Tensor<T>::Tensor(std::vector<T> data, std::vector<std::size_t> dims) : data_{std::move(data)}, view_{data_, dims} {}

template <IsAnyOf<double, std::complex<double>> T>
Tensor<T>& Tensor<T>::reshape(std::vector<std::size_t> dims) {
  view_.reshape(std::move(dims));
  return *this;
}

template <IsAnyOf<double, std::complex<double>> T>
Tensor<std::complex<double>> Tensor<T>::fft() const {
  Tensor<std::complex<double>> out{view_.dimensions()};
  view_.fft(out.view_);
  return out;
}

template <IsAnyOf<double, std::complex<double>> T>
Tensor<double> Tensor<T>::ifft() const {
  Tensor<double> out{view_.dimensions()};
  view_.ifft(out.view_);
  return out;
}

template class Tensor<double>;
template class Tensor<std::complex<double>>;

}  // namespace lucid