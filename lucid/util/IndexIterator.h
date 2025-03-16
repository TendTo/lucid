/**
 * @author c3054737
 * @copyright 2025 keid
 * @licence BSD 3-Clause License
 * @file
 * IndexIterator class.
 */
#pragma once

#include <iosfwd>
#include <ranges>
#include <vector>

#include "lucid/util/concept.h"

namespace lucid {
/**
 * Iterator over all possible indexes in a given range.
 * It also supports individual range for each index.
 * @code
 * for (IndexIterator<long> it{3, 0, 2}; it; ++it) {
 *  std::cout << it[0] << it[1] << it[2] << std::endl;
 * }
 * for (IndexIterator<std::vector<long>> it{{0, -1, 0}, {4, 2, 2}}; it; ++it) {
 *    std::cout << it[0] << it[1] << it[2] << std::endl;
 * }
 * @endcode
 * @tparam T type of the range. It can be a single value or a vector of values.
 */
template <IsAnyOf<long, std::vector<long>> T>
class IndexIterator {
 public:
  /**
   * Construct an index iterator with `size` given by `max_value.size()`.
   * Each of the indexes will go from [0 to `max_value[i]` - 1] (inclusive).
   * @param max_value maximum value for each index
   */
  explicit IndexIterator(std::vector<long> max_value)
    requires std::is_same_v<T, std::vector<long>>;
  /**
   * Construct an index iterator with `size` given by `max_value.size()`.
   * Each of the indexes will go from [`min_value[i]` to `max_value[i]` - 1] (inclusive).
   * @param min_value minimum value for each index
   * @param max_value maximum value for each index
   */
  IndexIterator(std::vector<long> min_value, std::vector<long> max_value)
    requires std::is_same_v<T, std::vector<long>>;
  /**
   * Construct an index iterator with the given `size`.
   * Each of the indexes will go from [0 to `max_value` - 1] (inclusive).
   * @param size number of indexes
   * @param max_value maximum value for each index
   */
  IndexIterator(std::size_t size, long max_value)
    requires std::is_same_v<T, long>;
  /**
   * Construct an index iterator with the given `size`.
   * Each of the indexes will go from [`min_value` to `max_value` - 1] (inclusive).
   * @param size number of indexes
   * @param min_value minimum value for each index
   * @param max_value maximum value for each index
   */
  IndexIterator(std::size_t size, long min_value, long max_value)
    requires std::is_same_v<T, long>;

  /**
   * Increment the iterator.
   * Go to the next index.
   * @return reference to the iterator
   */
  IndexIterator& operator++();
  /** @getter{whole vector of indexes, index iterator} */
  [[nodiscard]] const std::vector<long>& indexes() const { return indexes_; }
  /** @getter{index element, index iterator} */
  [[nodiscard]] long operator[](const std::size_t index) const { return indexes_[index]; }

  /** @checker{done iterating\, having gone over all valid indexes, index iterator} */
  operator bool() const;

 private:
  T min_value_;                ///< Minimum value for each index. Inclusive
  T max_value_;                ///< Maximum value for each index. Exclusive
  std::vector<long> indexes_;  ///< Current indexes
};

template <IsAnyOf<long, std::vector<long>> T>
std::ostream& operator<<(std::ostream& os, const IndexIterator<T>& index_iterator);

extern template class IndexIterator<long>;
extern template class IndexIterator<std::vector<long>>;

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::IndexIterator<long>)

#endif
