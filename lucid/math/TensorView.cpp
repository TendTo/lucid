/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/TensorView.h"

#include <ostream>

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
void TensorView<T>::fft(TensorView<std::complex<double>>& out, const double coeff) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(dims_ == out.dims_, "out.dimensions()", out.dims_, dims_);

  // Compute strides on complex data. They are the same for input and output
  std::vector<Index> strides{strides_};
  for (Index& i : strides) i *= sizeof(std::complex<double>);
  // The pointer to the input data only accepts complex data. If the input is not complex, we must first cast it
  std::complex<double>* in_data;
  std::vector<std::complex<double>> cast_vector;  // Only used as temporary storage if the input is not complex
  // Output data is always complex. Get the pointer where to store the result
  auto* const out_data = const_cast<std::complex<double>*>(out.data_.data());

  if constexpr (std::is_same_v<T, std::complex<double>>) {  // If the input was already complex, no need to cast it
    in_data = const_cast<T*>(data_.data());
  } else {  // Otherwise, cast it to complex
    cast_vector.reserve(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) cast_vector.push_back(static_cast<std::complex<double>>(data_[i]));
    in_data = cast_vector.data();
  }

  pocketfft::c2c(dims_, strides, strides, axes_, pocketfft::FORWARD, in_data, out_data, std::isnan(coeff) ? 1. : coeff);
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
void TensorView<T>::ifft(TensorView<std::complex<double>>& out, const double coeff) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(dims_ == out.dims_, "out.dimensions()", out.dims_, dims_);

  // Compute strides on complex data. They are the same for input and output
  std::vector<Index> strides{strides_};
  for (Index& i : strides) i *= sizeof(std::complex<double>);
  // The pointer to the input data only accepts complex data. If the input is not complex, we must first cast it
  std::complex<double>* in_data;
  std::vector<std::complex<double>> cast_vector;  // Only used as temporary storage if the input is not complex
  // Output data is always complex. Get the pointer where to store the result
  auto* const out_data = const_cast<std::complex<double>*>(out.data_.data());

  if constexpr (std::is_same_v<T, std::complex<double>>) {  // If the input was already complex, no need to cast it
    in_data = const_cast<T*>(data_.data());
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
  ifft(const_cast<TensorView<std::complex<double>>&>(out_complex.view()), coeff);
  std::span<double> out_data{const_cast<double*>(out.data_.data()), out.data_.size()};
  for (std::size_t i = 0; i < out_data.size(); ++i) {
    out_data[i] = out_complex[i].real();
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
  LUCID_CHECK_ARGUMENT(
      std::ranges::all_of(padding, [&out, this](const auto& p) { return p.first >= 0 && p.second >= 0; }), "padding",
      "must be non-negative");
  LUCID_CHECK_ARGUMENT(std::ranges::all_of(std::views::iota(static_cast<std::size_t>(0), dims_.size() - 1),
                                           [padding, &out, this](std::size_t i) {
                                             return dims_[i] + padding[i].first + padding[i].second == out.dims_[i];
                                           }),
                       "out", "dimensions must be equal to input tensor dimensions plus padding");
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

  std::span<T> out_data{const_cast<T*>(out.data_.data()), out.data_.size()};
  std::vector<long> max_idx(dims_.begin(), dims_.end());
  max_idx.back() = 1;
  for (IndexIterator<std::vector<long>> it{std::move(max_idx)}; it; ++it) {
    std::vector<long> output_idx{it.indexes()};
    for (std::size_t i = 0; i < dims_.size() - 1; i++) {
      if (output_idx[i] >= start_padding[i]) output_idx[i] += padding[i];
    }
    const std::span<const T> in_data_start_half = data_.subspan(index(std::span<const long>{it}), start_padding.back());
    const std::span<const T> in_data_end_half =
        data_.subspan(index(std::span<const long>{it}) + start_padding.back(), dims_.back() - start_padding.back());
    std::copy(in_data_start_half.begin(), in_data_start_half.end(),
              out_data.subspan(out.index(std::span<const long>{output_idx})).begin());
    std::copy(
        in_data_end_half.begin(), in_data_end_half.end(),
        out_data.subspan(out.index(std::span<const long>{output_idx})).begin() + padding.back() + start_padding.back());
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
  fft(const_cast<TensorView<std::complex<double>>&>(fft_out.view()));

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
  pad_out.view().ifft(out);

  Tensor<std::complex<double>> ifft_out{out.dims_};
  pad_out.view().ifft(const_cast<TensorView<std::complex<double>>&>(ifft_out.view()));
  Tensor<double> temp{out.dims_};
  {
    std::span<double> out_data{const_cast<double*>(temp.view().data_.data()), temp.view().data_.size()};
    for (std::size_t i = 0; i < out.size(); ++i) {
      out_data[i] = ifft_out[i].real() * pow(static_cast<double>(out.dims_[0]) / static_cast<double>(dims_[0]), 2);
    }
  }

  // Correct the scaling, since only the original tensor size should be used. The padding does not add any information
  for (std::span<double> out_data{const_cast<double*>(out.data_.data()), out.data_.size()}; double& d : out_data) {
    for (std::size_t rank = 0; rank < dims_.size(); ++rank) {
      d *= static_cast<double>(out.dims_[rank]) / static_cast<double>(dims_[rank]);
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
TensorView<T>& TensorView<T>::permute(const std::vector<std::size_t>& permutation) {
  LUCID_CHECK_ARGUMENT(std::ranges::all_of(permutation, [this](const std::size_t i) { return i < dims_.size(); }),
                       "permutation values", "must not exceed the rank of the tensor");
  std::vector<std::size_t> permuted_axes{axes_}, permuted_dims{dims_};
  std::vector<Index> permuted_strides{strides_};
  for (std::size_t i = 0; i < permutation.size() && i < dims_.size(); ++i) {
    permuted_axes[i] = permutation[i];
    permuted_dims[i] = dims_[permutation[i]];
    permuted_strides[i] = strides_[permutation[i]];
  }
  axes_ = std::move(permuted_axes);
  dims_ = std::move(permuted_dims);
  strides_ = std::move(permuted_strides);
  return *this;
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
std::ostream& operator<<(std::ostream& os, const TensorView<T>& tensor) {
  std::cout << "[ ";
  for (const auto& dim : tensor.dimensions()) std::cout << dim << " ";
  std::cout << "]\n";

  if (tensor.rank() == 1) {
    return os << Eigen::Map<const Eigen::RowVectorX<T>>{tensor.data().data(), static_cast<Index>(tensor.size())};
  }
  if (tensor.rank() == 2) {
    return os << Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>{
               tensor.data().data(), static_cast<Index>(tensor.dimensions()[0]),
               static_cast<Index>(tensor.dimensions()[1])};
  }
  for (std::size_t i = 0; i < tensor.size(); ++i) {
    std::cout << tensor.data()[i] << " ";
    if ((i + 1) % tensor.dimensions().back() == 0) {
      std::cout << "\n";
    }
  }
  return os;
}

template class TensorView<int>;
template class TensorView<float>;
template class TensorView<double>;
template class TensorView<std::complex<double>>;
template std::ostream& operator<<(std::ostream& os, const TensorView<int>& tensor);
template std::ostream& operator<<(std::ostream& os, const TensorView<float>& tensor);
template std::ostream& operator<<(std::ostream& os, const TensorView<double>& tensor);
template std::ostream& operator<<(std::ostream& os, const TensorView<std::complex<double>>& tensor);

}  // namespace lucid
