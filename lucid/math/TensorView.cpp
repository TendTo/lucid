/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/TensorView.h"

#include <ostream>

#include "lucid/lib/pocketfft.h"
#include "lucid/util/IndexIterator.h"
#include "lucid/util/error.h"

namespace lucid {

template <IsAnyOf<double, std::complex<double>> T>
TensorView<T>::TensorView(std::span<const T> data, std::vector<std::size_t> dims)
    : data_{std::move(data)}, dims_{std::move(dims)}, axes_(dims_.size()), strides_(dims_.size()) {
  if (dims_.empty()) {
    if (!data_.empty()) LUCID_INVALID_ARGUMENT_EXPECTED("data", data_.size(), "empty");
    return;
  }
  if (std::accumulate(dims_.begin(), dims_.end(), static_cast<std::size_t>(1), std::multiplies{}) != data_.size()) {
    LUCID_INVALID_ARGUMENT_EXPECTED(
        "data size", std::accumulate(dims_.begin(), dims_.end(), static_cast<std::size_t>(1), std::multiplies{}),
        data_.size());
  }
  strides_.back() = 1;
  for (std::size_t i = 1; i < dims_.size(); i++)
    strides_[strides_.size() - i - 1] = (strides_[strides_.size() - i] * dims_[dims_.size() - i]);
  std::iota(axes_.begin(), axes_.end(), 0);
}

template <IsAnyOf<double, std::complex<double>> T>
void TensorView<T>::fft(TensorView<std::complex<double>>& out) const {
  LUCID_ASSERT(dims_ == out.dims_, "Output tensor must have the same dimensions as the input tensor");
  T* const in_data = const_cast<T*>(data_.data());
  auto* const out_data = const_cast<std::complex<double>*>(out.data_.data());
  std::vector<Index> strides{strides_};
  for (Index& i : strides) i *= sizeof(T);
  if constexpr (std::is_same_v<T, std::complex<double>>) {
    for (std::size_t axis = 0; axis < dims_.size(); axis++) {
      pocketfft::c2c(dims_, strides, strides, axes_, pocketfft::FORWARD, in_data, out_data, 1.);
    }
  } else if constexpr (std::is_same_v<T, double>) {
    std::vector<Index> strides_out{strides_};
    for (Index& i : strides_out) i *= sizeof(std::complex<double>);
    pocketfft::r2c(dims_, strides, strides_out, axes_, pocketfft::FORWARD, in_data, out_data, 1.);
  } else {
    LUCID_NOT_SUPPORTED("Tensor which is not double or std::complex<double>");
  }
}

template <IsAnyOf<double, std::complex<double>> T>
void TensorView<T>::ifft(TensorView<double>& out) const {
  LUCID_ASSERT(dims_ == out.dims_, "Output tensor must have the same dimensions as the input tensor");
  T* const in_data = const_cast<T*>(data_.data());
  auto* const out_data = const_cast<double*>(out.data_.data());
  std::vector<Index> strides_in{strides_};
  for (Index& i : strides_in) i *= sizeof(T);
  std::vector<Index> strides_out{strides_};
  for (Index& i : strides_out) i *= sizeof(double);
  if constexpr (std::is_same_v<T, std::complex<double>>) {
    pocketfft::c2r(dims_, strides_in, strides_out, axes_, pocketfft::BACKWARD, in_data, out_data, 1. / size());
  } else {
    LUCID_NOT_SUPPORTED("Tensor which is not double or std::complex<double>");
  }
}
template <IsAnyOf<double, std::complex<double>> T>
TensorView<T>::operator Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>() const {
  if (dims_.size() != 2) LUCID_NOT_SUPPORTED("Only 2D tensors are supported. Use reshape to convert to 2D");
  return {data_.data(), static_cast<Index>(dims_.at(0)), static_cast<Index>(dims_.at(1))};
}

template <IsAnyOf<double, std::complex<double>> T>
void TensorView<T>::pad(TensorView<T>& out, const std::vector<std::pair<Index, Index>>& padding) const {
  if (std::ranges::any_of(padding, [](const std::pair<Index, Index>& p) { return p.first < 0 || p.second < 0; })) {
    LUCID_INVALID_ARGUMENT("Padding must be non-negative", padding);
  }
  LUCID_ASSERT(dims_.size() == out.dims_.size(), "Output tensor must have the same dimensions as the input tensor");
  LUCID_ASSERT(std::ranges::all_of(std::views::iota(static_cast<std::size_t>(0), dims_.size() - 1),
                                   [padding, &out, this](std::size_t i) {
                                     return dims_[i] + padding[i].first + padding[i].second == out.dims_[i];
                                   }),
               "Output tensor dimensions must be equal to input tensor dimensions plus padding");
  std::span<T> out_data{const_cast<T*>(out.data_.data()), out.data_.size()};
  std::vector<long> min_idx_out(dims_.size());
  std::vector<long> max_idx_out(dims_.begin(), dims_.end());
  std::vector<long> max_idx_in(dims_.begin(), dims_.end());
  max_idx_out.back() = 1;
  max_idx_in.back() = 1;
  for (std::size_t i = 0; i < dims_.size(); i++) {
    min_idx_out[i] = padding[i].first;
    max_idx_out[i] += padding[i].first;
  }
  IndexIterator<std::vector<long>> it_in{std::move(max_idx_in)};
  IndexIterator<std::vector<long>> it_out{std::move(min_idx_out), std::move(max_idx_out)};
  for (; it_out && it_in; ++it_out, ++it_in) {
    const std::span<const long> indexes_in{it_in};
    const std::span<const long> indexes_out{it_out};
    const std::span<const T> in_data = data_.subspan(index(indexes_in), dims_[dims_.size() - 1]);
    std::copy(in_data.begin(), in_data.end(), out_data.subspan(out.index(indexes_out)).begin());
  }
  LUCID_ASSERT(!it_in && !it_out, "Both iterators must have the same number of elements");
}

template <IsAnyOf<double, std::complex<double>> T>
TensorView<T>& TensorView<T>::reshape(std::vector<std::size_t> dims) {
  if (std::accumulate(dims.begin(), dims.end(), static_cast<std::size_t>(1), std::multiplies{}) != size())
    LUCID_INVALID_ARGUMENT_EXPECTED(
        "new size", std::accumulate(dims.begin(), dims.end(), static_cast<std::size_t>(1), std::multiplies{}), size());
  dims_ = std::move(dims);
  strides_.resize(dims_.size());
  strides_.back() = 1;
  for (std::size_t i = 1; i < dims_.size(); i++)
    strides_[strides_.size() - i - 1] = (strides_[strides_.size() - i] * dims_[dims_.size() - i]);
  return *this;
}

template <IsAnyOf<double, std::complex<double>> T>
std::ostream& operator<<(std::ostream& os, const TensorView<T>& tensor) {
  std::cout << "[ ";
  for (const auto& dim : tensor.dimensions()) std::cout << dim << " ";
  std::cout << "]\n";

  for (std::size_t i = 0; i < tensor.size(); ++i) {
    std::cout << tensor.data()[i] << " ";
    if ((i + 1) % tensor.dimensions().back() == 0) {
      std::cout << "\n";
    }
  }
  return os;
}

template class TensorView<double>;
template class TensorView<std::complex<double>>;
template std::ostream& operator<<(std::ostream& os, const TensorView<double>& tensor);
template std::ostream& operator<<(std::ostream& os, const TensorView<std::complex<double>>& tensor);

}  // namespace lucid
