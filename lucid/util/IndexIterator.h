/**
 * @author lucid_authors
 * @copyright 2025 keid
 * @licence BSD 3-Clause License
 * @file
 * IndexIterator class.
 */
#pragma once

#include <iosfwd>
#include <ranges>
#include <span>
#include <vector>

#include "lucid/lib/eigen.h"
#include "lucid/util/concept.h"

namespace lucid {
/**
 * Iterator over all possible indexes in a given range.
 * It also supports individual range for each index.
 * @code
 * for (IndexIterator<Index> it{3, 0, 2}; it; ++it) {
 *  std::cout << it[0] << it[1] << it[2] << std::endl;
 * }
 * for (IndexIterator<std::vector<Index>> it{{0, -1, 0}, {4, 2, 2}}; it; ++it) {
 *    std::cout << it[0] << it[1] << it[2] << std::endl;
 * }
 * @endcode
 * @tparam T type of the range. It can be a single value or a vector of values.
 */
template <IsAnyOf<Index, std::vector<Index>> T>
class IndexIterator {
 public:
  /**
   * Create an exhausted index iterator by assigning the same value to both min and max: 0.
   * Useful to create something cheap to indicate that there is nothing to iterate.
   * @return index iterator already exhausted
   */
  static IndexIterator<T> end();

  /**
   * Construct an index iterator with `size` given by `max_value.size()`.
   * Each of the indexes will go from [0 to `max_value[i]` - 1] (inclusive).
   * @param max_value maximum value for each index
   */
  explicit IndexIterator(T max_value)
    requires std::is_same_v<T, std::vector<Index>>;
  /**
   * Construct an index iterator with `size` given by `max_value.size()`.
   * Each of the indexes will go from [`min_value[i]` to `max_value[i]` - 1] (inclusive).
   * @param min_value minimum value for each index
   * @param max_value maximum value for each index
   */
  IndexIterator(T min_value, T max_value)
    requires std::is_same_v<T, std::vector<Index>>;
  /**
   * Construct an index iterator with the given `size`.
   * Each of the indexes will go from [0 to `max_value` - 1] (inclusive).
   * @param size number of indexes
   * @param max_value maximum value for each index
   */
  IndexIterator(std::size_t size, T max_value)
    requires std::is_same_v<T, Index>;
  /**
   * Construct an index iterator with the given `size`.
   * Each of the indexes will go from [`min_value` to `max_value` - 1] (inclusive).
   * @param size number of indexes
   * @param min_value minimum value for each index
   * @param max_value maximum value for each index
   */
  IndexIterator(std::size_t size, T min_value, T max_value)
    requires std::is_same_v<T, Index>;

  /**
   * Increment the iterator.
   * Go to the next index.
   * @return reference to the iterator
   */
  IndexIterator& operator++();
  /** @getter{whole vector of indexes, index iterator} */
  [[nodiscard]] const std::vector<Index>& indexes() const { return indexes_; }
  /** @getter{index element, index iterator} */
  [[nodiscard]] Index operator[](const std::size_t index) const { return indexes_[index]; }

  /** @checker{done iterating\, having gone over all valid indexes, index iterator} */
  operator bool() const;
  operator std::span<const Index>() const;

 private:
  T min_value_;                 ///< Minimum value for each index. Inclusive
  T max_value_;                 ///< Maximum value for each index. Exclusive
  std::vector<Index> indexes_;  ///< Current indexes
};

template <IsAnyOf<Index, std::vector<Index>> T>
std::ostream& operator<<(std::ostream& os, const IndexIterator<T>& index_iterator);

extern template class IndexIterator<Index>;
extern template class IndexIterator<std::vector<Index>>;

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::IndexIterator<lucid::Index>)
OSTREAM_FORMATTER(lucid::IndexIterator<std::vector<lucid::Index>>)

#endif
