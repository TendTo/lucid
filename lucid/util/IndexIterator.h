/**
 * @author c3054737
 * @copyright 2025 keid
 * @licence BSD 3-Clause License
 * @file
 * IndexIterator class.
 */
#pragma once

#include <iostream>
#include <ranges>
#include <vector>

namespace lucid {

/**
 * Iterator over all possible indexes in a given range.
 * @code
 * for (IndexIterator it{3, 0, 2}; it; ++it) {
 *  std::cout << it[0] << it[1] << it[2] << std::endl;
 * }
 * @endcode
 */
class IndexIterator {
 public:
  /**
   * Construct an index iterator with the given `size`.
   * Each of the indexes will go from [0 to `max_value` - 1] (inclusive).
   * @param size number of indexes
   * @param max_value maximum value for each index
   */
  IndexIterator(const std::size_t size, const long max_value) : IndexIterator{size, 0, max_value} {}
  /**
   * Construct an index iterator with the given `size`.
   * Each of the indexes will go from [`min_value` to `max_value` - 1] (inclusive).
   * @param size number of indexes
   * @param min_value minimum value for each index
   * @param max_value maximum value for each index
   */
  IndexIterator(std::size_t size, long min_value, long max_value);

  IndexIterator& operator++() {
    indexes_.back()++;
    if (indexes_.back() < max_value_) return *this;
    for (std::size_t i = indexes_.size() - 1; i > 0; i--) {
      if (indexes_[i] >= max_value_) {
        indexes_[i] = min_value_;
        indexes_[i - 1]++;
      }
    }
    return *this;
  }

  [[nodiscard]] const std::vector<long>& indexes() const { return indexes_; }

  [[nodiscard]] long operator[](const std::size_t index) const { return indexes_[index]; }

  operator bool() const { return indexes_.front() < max_value_; }

 private:
  long min_value_;             ///< Minimum value for each index. Inclusive
  long max_value_;             ///< Maximum value for each index. Exclusive
  std::vector<long> indexes_;  ///< Current indexes
};

}  // namespace lucid
