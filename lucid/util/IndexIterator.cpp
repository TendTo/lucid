/**
 * @author c3054737
 * @copyright 2025 keid
 * @licence BSD 3-Clause License
 * @file
 * IndexIterator class.
 */
#include "lucid/util/IndexIterator.h"

#include <ostream>
#include <utility>

#include "lucid/util/error.h"

namespace lucid {

template <IsAnyOf<long, std::vector<long>> T>
IndexIterator<T>::IndexIterator(const std::size_t size, long min_value, long max_value)
  requires std::is_same_v<T, long>
    : min_value_{min_value}, max_value_{max_value}, indexes_(size, min_value) {
  if (size == 0) LUCID_INVALID_ARGUMENT_EXPECTED("size", size, "greater than 0");
  if (min_value_ > max_value_) LUCID_INVALID_ARGUMENT_EXPECTED("min_value", min_value, "less than max_value");
}

template <IsAnyOf<long, std::vector<long>> T>
IndexIterator<T>::IndexIterator(std::vector<long> max_value)
  requires std::is_same_v<T, std::vector<long>>
    : IndexIterator{std::vector<long>(max_value.size(), 0), std::move(max_value)} {}

template <IsAnyOf<long, std::vector<long>> T>
IndexIterator<T>::IndexIterator(std::vector<long> min_value, std::vector<long> max_value)
  requires std::is_same_v<T, std::vector<long>>
    : min_value_{std::move(min_value)}, max_value_{std::move(max_value)}, indexes_{min_value_} {
  if (min_value_.empty()) LUCID_INVALID_ARGUMENT_EXPECTED("min_value.size()", min_value_.size(), "greater than 0");
  if (min_value_.size() != max_value_.size()) {
    LUCID_INVALID_ARGUMENT_EXPECTED("min_value.size()", min_value_.size(), max_value_.size());
  }
  for (std::size_t i = 0; i < min_value_.size(); i++) {
    if (min_value_[i] >= max_value_[i]) {
      LUCID_INVALID_ARGUMENT_EXPECTED("min_value < max_value", min_value_[i], max_value_[i] - 1);
    }
  }
}

template <IsAnyOf<long, std::vector<long>> T>
IndexIterator<T>::IndexIterator(std::size_t size, long max_value)
  requires std::is_same_v<T, long>
    : IndexIterator{size, 0, max_value} {}

template <IsAnyOf<long, std::vector<long>> T>
IndexIterator<T>& IndexIterator<T>::operator++() {
  indexes_.back()++;
  if constexpr (std::is_same_v<T, long>) {
    if (indexes_.back() < max_value_) return *this;
    for (std::size_t i = indexes_.size() - 1; i > 0; i--) {
      if (indexes_[i] >= max_value_) {
        indexes_[i] = min_value_;
        indexes_[i - 1]++;
      }
    }
  } else {
    if (indexes_.back() < max_value_.back()) return *this;
    for (std::size_t i = indexes_.size() - 1; i > 0; i--) {
      if (indexes_[i] >= max_value_[i]) {
        indexes_[i] = min_value_[i];
        indexes_[i - 1]++;
      }
    }
  }
  return *this;
}

template <IsAnyOf<long, std::vector<long>> T>
IndexIterator<T>::operator bool() const {
  if constexpr (std::is_same_v<T, long>) {
    return indexes_.front() < max_value_;
  } else {
    return indexes_.front() < max_value_.front();
  }
}
template <IsAnyOf<long, std::vector<long>> T>
IndexIterator<T>::operator std::span<const long>() const {
  return std::span<const long>{indexes_};
}

template <IsAnyOf<long, std::vector<long>> T>
std::ostream& operator<<(std::ostream& os, const IndexIterator<T>& index_iterator) {
  os << '[';
  for (std::size_t i = 0; i < index_iterator.indexes().size(); i++) {
    os << index_iterator[i];
    if (i < index_iterator.indexes().size() - 1) os << ", ";
  }
  return os << ']';
}

template class IndexIterator<long>;
template class IndexIterator<std::vector<long>>;
template std::ostream& operator<<(std::ostream& os, const IndexIterator<long>& index_iterator);
template std::ostream& operator<<(std::ostream& os, const IndexIterator<std::vector<long>>& index_iterator);

}  // namespace lucid