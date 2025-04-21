/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/TensorView.h"

#include <algorithm>
#include <ostream>
#include <set>
#include <utility>
#include <vector>

#include "lucid/lib/pocketfft.h"
#include "lucid/math/Tensor.h"
#include "lucid/util/IndexIterator.h"
#include "lucid/util/error.h"

namespace lucid {

template <IsAnyOf<int, float, double, std::complex<double>> T>
TensorView<T>::TensorView(std::span<const T> data, std::vector<std::size_t> dims)
    : data_{std::move(data)}, dims_{std::move(dims)}, axes_(dims_.size()), strides_(dims_.size()) {
  if (dims_.empty()) {
    LUCID_CHECK_ARGUMENT(data_.empty(), "data", "must be empty if dims is empty");
    return;
  }
  LUCID_CHECK_ARGUMENT(
      std::accumulate(dims_.begin(), dims_.end(), static_cast<std::size_t>(1), std::multiplies{}) == data_.size(),
      "data", "size must be equal to the product of the dimensions");
  strides_.back() = 1;
  for (std::size_t i = 1; i < dims_.size(); i++)
    strides_[strides_.size() - i - 1] = (strides_[strides_.size() - i] * dims_[dims_.size() - i]);
  std::iota(axes_.begin(), axes_.end(), 0);
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
TensorIterator<T> TensorView<T>::begin() const {
  return TensorIterator<T>{*this};
}
template <IsAnyOf<int, float, double, std::complex<double>> T>
TensorIterator<T> TensorView<T>::end() const {
  return TensorIterator<T>{*this, true};
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
void TensorView<T>::fft(TensorView<std::complex<double>>& out, const double coeff) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(out.size() == size(), "out.size()", out.size(), size());

  // Set output dimensions and strides
  out.dims_ = dims_;
  out.strides_ = strides_;
  out.axes_ = axes_;

  // Compute strides on complex data. They are the same for input and output
  std::vector<Index> strides{strides_};
  for (Index& i : strides) i *= sizeof(std::complex<double>);
  // The pointer to the input data only accepts complex data. If the input is not complex, we must first cast it
  const std::complex<double>* in_data;
  std::vector<std::complex<double>> cast_vector;  // Only used as temporary storage if the input is not complex
  // Output data is always complex. Get the pointer where to store the result
  std::complex<double>* const out_data = out.m_data().data();

  if constexpr (std::is_same_v<T, std::complex<double>>) {  // If the input was already complex, no need to cast it
    in_data = data_.data();
  } else {  // Otherwise, cast it to complex
    cast_vector.reserve(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) cast_vector.push_back(static_cast<std::complex<double>>(data_[i]));
    in_data = cast_vector.data();
  }

  pocketfft::c2c(dims_, strides, strides, axes_, pocketfft::FORWARD, in_data, out_data, std::isnan(coeff) ? 1. : coeff);
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
void TensorView<T>::ifft(TensorView<std::complex<double>>& out, const double coeff) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(out.size() == size(), "out.size()", out.size(), size());

  // Set output dimensions and strides
  out.dims_ = dims_;
  out.strides_ = strides_;
  out.axes_ = axes_;

  // Compute strides on complex data. They are the same for both input and output
  std::vector<Index> strides{strides_};
  for (Index& i : strides) i *= sizeof(std::complex<double>);
  // The pointer to the input data only accepts complex data. If the input is not complex, we must first cast it
  const std::complex<double>* in_data;
  std::vector<std::complex<double>> cast_vector;  // Only used as temporary storage if the input is not complex
  // Output data is always complex. Get the pointer where to store the result
  std::complex<double>* const out_data = out.m_data().data();

  if constexpr (std::is_same_v<T, std::complex<double>>) {  // If the input was already complex, no need to cast it
    in_data = data_.data();
  } else {  // Otherwise, cast it to complex
    cast_vector.reserve(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) cast_vector.push_back(static_cast<std::complex<double>>(data_[i]));
    in_data = cast_vector.data();
  }

  pocketfft::c2c(dims_, strides, strides, axes_, pocketfft::BACKWARD, in_data, out_data,
                 std::isnan(coeff) ? 1. / size() : coeff);
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
void TensorView<T>::ifft(TensorView<double>& out, const double coeff) const {
  Tensor<std::complex<double>> out_complex{out.dims_};
  ifft(out_complex.m_view(), coeff);
  for (std::size_t i = 0; i < out.data_.size(); ++i) {
    out.m_data()[i] = out_complex[i].real();
  }
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
TensorView<T>::operator Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>() const {
  if (dims_.size() != 2) LUCID_NOT_SUPPORTED("Only 2D tensors are supported. Use reshape to convert to 2D");
  return {data_.data(), static_cast<Index>(dims_.at(0)), static_cast<Index>(dims_.at(1))};
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
void TensorView<T>::pad(TensorView<T>& out, const std::vector<std::pair<Index, Index>>& padding) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(out.rank() == rank(), "out.rank()", out.rank(), rank());
  LUCID_CHECK_ARGUMENT(std::ranges::all_of(padding, [](const auto& p) { return p.first >= 0 && p.second >= 0; }),
                       "padding", "must be non-negative");
  LUCID_CHECK_ARGUMENT(std::ranges::all_of(std::views::iota(static_cast<std::size_t>(0), dims_.size() - 1),
                                           [padding, &out, this](std::size_t i) {
                                             return dims_[i] + padding[i].first + padding[i].second == out.dims_[i];
                                           }),
                       "out", "dimensions must be equal to input tensor dimensions plus padding");
  std::vector<Index> min_idx_out(dims_.size());
  std::vector<Index> max_idx_out(dims_.begin(), dims_.end());
  std::vector<Index> max_idx_in(dims_.begin(), dims_.end());
  max_idx_out.back() = 1;
  max_idx_in.back() = 1;
  for (std::size_t i = 0; i < dims_.size(); i++) {
    min_idx_out[i] = padding[i].first;
    max_idx_out[i] += padding[i].first;
  }
  IndexIterator<std::vector<Index>> it_in{std::move(max_idx_in)};
  IndexIterator<std::vector<Index>> it_out{std::move(min_idx_out), std::move(max_idx_out)};
  for (; it_out && it_in; ++it_out, ++it_in) {
    const std::span<const Index> indexes_in{it_in};
    const std::span<const Index> indexes_out{it_out};
    const std::span<const T> in_data = data_.subspan(index(indexes_in), dims_[dims_.size() - 1]);
    std::copy(in_data.begin(), in_data.end(), out.m_data().subspan(out.index(indexes_out)).begin());
  }
  LUCID_ASSERT(!it_in && !it_out, "Both iterators must have the same number of elements");
}
template <IsAnyOf<int, float, double, std::complex<double>> T>
void TensorView<T>::pad(TensorView<T>& out, const std::vector<Index>& padding,
                        const std::vector<Index>& start_padding) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(padding.size() == rank(), "padding.size()", padding.size(), rank());
  LUCID_CHECK_ARGUMENT_EXPECTED(out.rank() == rank(), "out.rank()", out.rank(), rank());
  LUCID_CHECK_ARGUMENT_EXPECTED(start_padding.size() == rank(), "start_padding.size()", start_padding.size(), rank());
  LUCID_CHECK_ARGUMENT_EXPECTED(
      std::ranges::all_of(std::views::iota(static_cast<std::size_t>(0), dims_.size() - 1),
                          [&padding, &out, this](std::size_t i) { return dims_[i] + padding[i] == out.dims_[i]; }),
      "out.dimensions()", out.dimensions(), "input.dimensions() + padding");
  LUCID_CHECK_ARGUMENT(std::ranges::all_of(padding, [](const Index p) { return p >= 0; }), "padding",
                       "must be non-negative");
  LUCID_CHECK_ARGUMENT(std::ranges::all_of(start_padding, [](const Index sp) { return sp >= 0; }), "start_padding",
                       "must be non-negative");
  LUCID_CHECK_ARGUMENT(std::ranges::all_of(std::views::iota(static_cast<std::size_t>(0), dims_.size() - 1),
                                           [&start_padding, this](const std::size_t i) {
                                             return static_cast<std::size_t>(start_padding.at(i)) <= dims_.at(i);
                                           }),
                       "start_padding", "must be less than or equal to input tensor dimensions");

  std::vector<Index> max_idx(dims_.begin(), dims_.end());
  max_idx.back() = 1;
  for (IndexIterator<std::vector<Index>> it{std::move(max_idx)}; it; ++it) {
    std::vector<Index> output_idx{it.indexes()};
    for (std::size_t i = 0; i < dims_.size() - 1; i++) {
      if (output_idx[i] >= start_padding[i]) output_idx[i] += padding[i];
    }
    const std::span<const T> in_data_start_half =
        data_.subspan(index(std::span<const Index>{it}), start_padding.back());
    const std::span<const T> in_data_end_half =
        data_.subspan(index(std::span<const Index>{it}) + start_padding.back(), dims_.back() - start_padding.back());
    std::copy(in_data_start_half.begin(), in_data_start_half.end(),
              out.m_data().subspan(out.index(std::span<const Index>{output_idx})).begin());
    std::copy(in_data_end_half.begin(), in_data_end_half.end(),
              out.m_data().subspan(out.index(std::span<const Index>{output_idx})).begin() + padding.back() +
                  start_padding.back());
  }
}
template <IsAnyOf<int, float, double, std::complex<double>> T>
void TensorView<T>::fft_upsample(TensorView<double>& out) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(out.rank() == rank(), "out.rank()", out.rank(), rank());
  LUCID_CHECK_ARGUMENT_EXPECTED(!dims_.empty(), "dimensions()", 0, "> 0");
  LUCID_CHECK_ARGUMENT_EXPECTED(!out.dims_.empty(), "out.dimensions()", 0, "> 0");
  LUCID_CHECK_ARGUMENT(
      std::ranges::all_of(std::views::iota(static_cast<std::size_t>(0), dims_.size() - 1),
                          [&out, this](const std::size_t i) { return out.dims_.at(i) >= dims_.at(i); }),
      "out.dimensions()", "must be greater than or equal to input tensor corresponding dimensions");
  LUCID_CHECK_ARGUMENT(std::ranges::all_of(dims_, [this](const std::size_t d) { return d == dims_[0]; }),
                       "dimensions()", "all dimensions must be equal");
  LUCID_CHECK_ARGUMENT(std::ranges::all_of(out.dims_, [&out](const std::size_t d) { return d == out.dims_[0]; }),
                       "out.dimensions()", "all dimensions must be equal");
  Tensor<std::complex<double>> fft_out{dims_};
  fft(fft_out.m_view());

  // Determine how much padding is needed and where it is going to be placed
  std::vector<Index> padding(dims_.size()), start_padding(dims_.size());
  for (std::size_t i = 0; i < dims_.size(); ++i) {
    padding[i] = out.dims_[i] - dims_[i];
    start_padding[i] = dims_[i] / 2 + 1;
  }

  Tensor<std::complex<double>> pad_out = fft_out.pad(padding, start_padding);

  // TODO(tend): Do we need this or not?? It does not seem to change anything in the interpolation, at least for reals
  // for (const std::size_t dim : dims_) {
  //   if (dim & 1) continue;
  //   const_cast<std::complex<double>*>(pad_out.view().data_.data())[dim / 2] /= 2;
  // }

  // Compute the inverse FFT of the padded tensor to interpolate the signal, resulting in an upsampled tensor
  // We need to adjust the scaling coefficient accordingly
  const double coeff = std::pow(1.0 / static_cast<double>(dims_[0]), rank());
  pad_out.view().ifft(out, coeff);

  // If we are dealing with a tensor where the dimensions have different sizes, we need to correct the scaling
  // NOLINTNEXTLINE(whitespace/newline): false positive
  if (std::ranges::any_of(dims_, [this](const std::size_t d) { return d != dims_[0]; })) {
    for (double& d : out.m_data()) {
      for (const std::size_t dim : dims_) d *= coeff / static_cast<double>(dim);
    }
  }
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
TensorView<T>& TensorView<T>::reshape(std::vector<std::size_t> dims) {
  LUCID_CHECK_ARGUMENT(
      std::accumulate(dims.begin(), dims.end(), static_cast<std::size_t>(1), std::multiplies{}) == size(), "new size",
      "must be equal to the current size");
  dims_ = std::move(dims);
  strides_.resize(dims_.size());
  strides_.back() = 1;
  for (std::size_t i = 1; i < dims_.size(); i++)
    strides_[strides_.size() - i - 1] = (strides_[strides_.size() - i] * dims_[dims_.size() - i]);
  return *this;
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
template <IsAnyOf<int, float, double, std::complex<double>> TT>
void TensorView<T>::permute(TensorView<TT>& out, const std::vector<std::size_t>& permutation) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(out.size() == size(), "out.size()", out.size(), size());
  LUCID_CHECK_ARGUMENT(std::ranges::all_of(permutation, [&](const std::size_t p) { return p < rank(); }),
                       "permutation values", "must be in [0, rank - 1]");

  // Create a temporary view to the input data, just so we can assert this method is const.
  // Its dimensions and strides get permuted
  TensorView<T> temp_view{data_, dims_};
  for (std::size_t i = 0; i < permutation.size() && i < dims_.size(); ++i) {
    temp_view.dims_[i] = dims_[permutation[i]];
    temp_view.strides_[i] = strides_[permutation[i]];
  }

  // Set output dimensions. Doing so will also update the strides accordingly
  out.reshape(temp_view.dims_);

  std::size_t i = 0;
  if constexpr (!std::is_same_v<T, TT> && std::is_same_v<T, std::complex<double>>) {
    for (const T& val : temp_view) out.m_data()[i++] = static_cast<TT>(val.real());
  } else {
    for (const T& val : temp_view) out.m_data()[i++] = static_cast<TT>(val);
  }
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
std::ostream& operator<<(std::ostream& os, const TensorView<T>& tensor) {
  os << "[ ";
  for (const auto& dim : tensor.dimensions()) os << dim << " ";
  os << "]\n";

  if (tensor.rank() == 1) {
    return os << static_cast<Eigen::Map<const Eigen::VectorX<T>>>(tensor);
  }
  if (tensor.rank() == 2) {
    return os << static_cast<Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>(
               tensor);
  }
  if (tensor.rank() == 3) {
    for (Index i = 0; i < static_cast<Index>(tensor.dimensions().back()); ++i) {
      os << "z: " << i << "\n";
      Eigen::MatrixX<T> vector{tensor.dimensions()[0], tensor.dimensions()[1]};
      for (IndexIterator<std::vector<Index>> it{
               {0, 0, i}, {static_cast<Index>(tensor.dimensions()[0]), static_cast<Index>(tensor.dimensions()[1]), i}};
           it; ++it) {
        vector(it.indexes()[0], it.indexes()[1]) = tensor(it.indexes());
      }
      os << vector << "\n";
    }
    return os;
  }
  std::size_t i = 0;
  for (const T& val : tensor) {
    os << val << " ";
    if ((i + 1) % tensor.dimensions().back() == 0) os << "\n";
    ++i;
  }
  return os;
}

template class TensorView<int>;
template class TensorView<float>;
template class TensorView<double>;
template class TensorView<std::complex<double>>;

template void TensorView<int>::permute(TensorView<int>&, const std::vector<std::size_t>&) const;
template void TensorView<float>::permute(TensorView<int>&, const std::vector<std::size_t>&) const;
template void TensorView<double>::permute(TensorView<int>&, const std::vector<std::size_t>&) const;
template void TensorView<std::complex<double>>::permute(TensorView<int>&, const std::vector<std::size_t>&) const;
template void TensorView<int>::permute(TensorView<float>&, const std::vector<std::size_t>&) const;
template void TensorView<float>::permute(TensorView<float>&, const std::vector<std::size_t>&) const;
template void TensorView<double>::permute(TensorView<float>&, const std::vector<std::size_t>&) const;
template void TensorView<std::complex<double>>::permute(TensorView<float>&, const std::vector<std::size_t>&) const;
template void TensorView<int>::permute(TensorView<double>&, const std::vector<std::size_t>&) const;
template void TensorView<float>::permute(TensorView<double>&, const std::vector<std::size_t>&) const;
template void TensorView<double>::permute(TensorView<double>&, const std::vector<std::size_t>&) const;
template void TensorView<std::complex<double>>::permute(TensorView<double>&, const std::vector<std::size_t>&) const;
template void TensorView<int>::permute(TensorView<std::complex<double>>&, const std::vector<std::size_t>&) const;
template void TensorView<float>::permute(TensorView<std::complex<double>>&, const std::vector<std::size_t>&) const;
template void TensorView<double>::permute(TensorView<std::complex<double>>&, const std::vector<std::size_t>&) const;
template void TensorView<std::complex<double>>::permute(TensorView<std::complex<double>>&,
                                                        const std::vector<std::size_t>&) const;

template std::ostream& operator<<(std::ostream& os, const TensorView<int>& tensor);
template std::ostream& operator<<(std::ostream& os, const TensorView<float>& tensor);
template std::ostream& operator<<(std::ostream& os, const TensorView<double>& tensor);
template std::ostream& operator<<(std::ostream& os, const TensorView<std::complex<double>>& tensor);

}  // namespace lucid
