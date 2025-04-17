/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * TensorIterator class.
 */
#pragma once

#include <complex>
#include <iosfwd>
#include <vector>

#include "lucid/util/IndexIterator.h"
#include "lucid/util/concept.h"

namespace lucid {

template <IsAnyOf<int, float, double, std::complex<double>> T>
class TensorView;

/**
 * Utility class to iterate over all elements of a tensor.
 * They will be visited in order, avoiding the need to use nested loops and hiding the memory layout and strides.
 * @tparam T type of the elements in the tensor
 */
template <IsAnyOf<int, float, double, std::complex<double>> T>
class TensorIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  /**
   * Construct a new TensorIterator object.
   * @param tensor tensor to iterate over
   * @param end if true, the iterator will be exhausted immediately, indicating that there is nothing else to iterate
   */
  explicit TensorIterator(const TensorView<T>& tensor, bool end = false);

  /** @getter{indexes, tensor} */
  [[nodiscard]] const std::vector<long>& indexes() const { return indexes_.indexes(); }
  /** @getter{index iterator, tensor} */
  [[nodiscard]] const IndexIterator<std::vector<long>>& index_iterator() const { return indexes_; }

  [[nodiscard]] const T& operator*() const;
  pointer operator->() const;
  TensorIterator& operator++();
  operator bool() const { return indexes_; }
  bool operator==(const TensorIterator<T>& o) const;

 private:
  IndexIterator<std::vector<long>> indexes_;  ///< Iterator used to go through all elements of the tensor
  const TensorView<T>& tensor_;               ///< Tensor to iterate over
};

template <IsAnyOf<int, float, double, std::complex<double>> T>
std::ostream& operator<<(std::ostream& os, const TensorIterator<T>& it);

extern template class TensorIterator<int>;
extern template class TensorIterator<float>;
extern template class TensorIterator<double>;
extern template class TensorIterator<std::complex<double>>;

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::TensorIterator<int>)
OSTREAM_FORMATTER(lucid::TensorIterator<float>)
OSTREAM_FORMATTER(lucid::TensorIterator<double>)
OSTREAM_FORMATTER(lucid::TensorIterator<std::complex<double>>)

#endif
