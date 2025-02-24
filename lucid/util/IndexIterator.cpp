/**
 * @author c3054737
 * @copyright 2025 keid
 * @licence BSD 3-Clause License
 * @file
 * IndexIterator class.
 */
#include "lucid/util/IndexIterator.h"

#include "lucid/util/error.h"

namespace lucid {

IndexIterator::IndexIterator(const std::size_t size, const long min_value, const long max_value)
    : min_value_{min_value}, max_value_{max_value}, indexes_(size, min_value) {
  if (size == 0) LUCID_INVALID_ARGUMENT_EXPECTED("size", size, "greater than 0");
  if (min_value > max_value) LUCID_INVALID_ARGUMENT_EXPECTED("min_value", min_value, "less than max_value");
}

}  // namespace lucid