/**
 * @author Room 6.030
 * @copyright 2025 keid
 * @licence BSD 3-Clause License
 * @file
 * IndexIterator class.
 */
#include "lucid/util/IndexIterator.h"

#include <ostream>
#include <utility>
#include <vector>

#include "lucid/util/error.h"

namespace lucid {

template <IsAnyOf<Index, std::vector<Index>> T>
IndexIterator<T> IndexIterator<T>::end() {
  if constexpr (std::is_same_v<T, Index>) {
    return {1, 0, 0};
  } else {
    return {std::vector<Index>{0}, std::vector<Index>{0}};
  }
}

template <IsAnyOf<Index, std::vector<Index>> T>
IndexIterator<T>::IndexIterator(std::size_t size, T max_value)
  requires std::is_same_v<T, Index>
    : IndexIterator{size, 0, max_value} {}

template <IsAnyOf<Index, std::vector<Index>> T>
IndexIterator<T>::IndexIterator(const std::size_t size, T min_value, T max_value)
  requires std::is_same_v<T, Index>
    : min_value_{min_value}, max_value_{max_value}, indexes_(size, min_value) {
  LUCID_CHECK_ARGUMENT_CMP(size, >, 0);
  LUCID_CHECK_ARGUMENT_CMP(min_value_, <=, max_value_);
}

// Using the move constructor for `max_value` creates issues in msvc++
template <IsAnyOf<Index, std::vector<Index>> T>
IndexIterator<T>::IndexIterator(T max_value)
  requires std::is_same_v<T, std::vector<Index>>
    : IndexIterator{std::vector<Index>(max_value.size(), 0), max_value} {}

template <IsAnyOf<Index, std::vector<Index>> T>
IndexIterator<T>::IndexIterator(T min_value, T max_value)
  requires std::is_same_v<T, std::vector<Index>>
    : min_value_{std::move(min_value)}, max_value_{std::move(max_value)}, indexes_{min_value_} {
  LUCID_CHECK_ARGUMENT_CMP(min_value_.size(), >, 0);
  LUCID_CHECK_ARGUMENT_EQ(min_value_.size(), max_value_.size());
#ifndef NCHECK
  for (std::size_t i = 0; i < min_value_.size(); i++) {
    if (min_value_[i] > max_value_[i]) {
      LUCID_INVALID_ARGUMENT_EXPECTED("min_value", min_value_[i], max_value_[i] - 1);
    }
  }
#endif
}

template <IsAnyOf<long int, std::vector<long>> T>
IndexIterator<T>& IndexIterator<T>::reset() {
  if constexpr (std::is_same_v<T, Index>) {
    indexes_.assign(indexes_.size(), min_value_);
  } else {
    indexes_ = min_value_;
  }
  return *this;
}

template <IsAnyOf<Index, std::vector<Index>> T>
IndexIterator<T>& IndexIterator<T>::operator++() {
  indexes_.back()++;
  if constexpr (std::is_same_v<T, Index>) {
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

template <IsAnyOf<Index, std::vector<Index>> T>
IndexIterator<T>::operator bool() const {
  if constexpr (std::is_same_v<T, Index>) {
    return indexes_.front() < max_value_;
  } else {
    return indexes_.front() < max_value_.front();
  }
}
template <IsAnyOf<Index, std::vector<Index>> T>
IndexIterator<T>::operator std::span<const Index>() const {
  return std::span<const Index>{indexes_};
}

template <IsAnyOf<Index, std::vector<Index>> T>
std::ostream& operator<<(std::ostream& os, const IndexIterator<T>& index_iterator) {
  os << '[';
  for (std::size_t i = 0; i < index_iterator.indexes().size(); i++) {
    os << index_iterator[i];
    if (i < index_iterator.indexes().size() - 1) os << ", ";
  }
  return os << ']';
}

template class IndexIterator<Index>;
template class IndexIterator<std::vector<Index>>;
template std::ostream& operator<<(std::ostream& os, const IndexIterator<Index>& index_iterator);
template std::ostream& operator<<(std::ostream& os, const IndexIterator<std::vector<Index>>& index_iterator);

}  // namespace lucid
