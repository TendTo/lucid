/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/Tensor.h"

#include "lucid/util/error.h"

namespace lucid {

template <IsAnyOf<int, float, double, std::complex<double>> T>
Tensor<T>::Tensor(std::vector<std::size_t> dims)
    : Tensor(std::vector<T>(std::accumulate(dims.begin(), dims.end(), static_cast<std::size_t>(1), std::multiplies{})),
             dims) {}
template <IsAnyOf<int, float, double, std::complex<double>> T>
Tensor<T>::Tensor(std::vector<T> data, std::vector<std::size_t> dims)
    : data_{std::move(data)}, view_{data_, dims.empty() ? std::vector<std::size_t>{data_.size()} : dims} {}
template <IsAnyOf<int, float, double, std::complex<double>> T>
Tensor<T>::Tensor(const T& value, std::vector<std::size_t> dims)
    : Tensor(std::vector<T>(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies{}), value), dims) {}

template <IsAnyOf<int, float, double, std::complex<double>> T>
Tensor<T>& Tensor<T>::reshape(std::vector<std::size_t> dims) {
  view_.reshape(std::move(dims));
  return *this;
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
Tensor<std::complex<double>> Tensor<T>::fft(const std::vector<std::size_t>& axes) const {
  Tensor<std::complex<double>> out{view_.dimensions()};
  view_.fft(out.view_, axes);
  return out;
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
Tensor<double> Tensor<T>::ifft(const std::vector<std::size_t>& axes) const {
  Tensor<double> out{view_.dimensions()};
  view_.ifft(out.view_, axes);
  return out;
}
template <IsAnyOf<int, float, double, std::complex<double>> T>
Tensor<double> Tensor<T>::fft_upsample(const std::vector<std::size_t>& new_dims,
                                       const std::vector<std::size_t>& axes) const {
  Tensor<double> out{new_dims};
  view_.fft_upsample(out.view_, axes);
  return out;
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
Tensor<T> Tensor<T>::pad(const std::vector<std::pair<Index, Index>>& padding, const T& value) const {
  std::vector<std::size_t> new_dims{view_.dimensions()};
  for (std::size_t i = 0; i < view_.rank(); ++i) new_dims[i] += padding[i].first + padding[i].second;
  Tensor<T> out{value, new_dims};
  view_.pad(out.view_, padding);
  return out;
}
template <IsAnyOf<int, float, double, std::complex<double>> T>
Tensor<T> Tensor<T>::pad(const std::vector<Index>& padding, const std::vector<Index>& start_padding,
                         const T& value) const {
  std::vector<std::size_t> new_dims{view_.dimensions()};
  for (std::size_t i = 0; i < view_.rank(); ++i) new_dims[i] += padding[i];
  Tensor<T> out{value, new_dims};
  view_.pad(out.view_, padding, start_padding);
  return out;
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
  return os << tensor.view();
}

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<std::complex<double>>;
template std::ostream& operator<<(std::ostream& os, const Tensor<int>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<float>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<double>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<std::complex<double>>& tensor);

}  // namespace lucid