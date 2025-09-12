/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Collection of metric measuring functions.
 */
#pragma once

#include <cstddef>
#include <iosfwd>

/**
 * @namespace lucid::metrics
 * Collection of metric measuring functions.
 */
namespace lucid::metrics {

/** Memory units for conversion. */
enum class MemoryUnit {
  B,   ///< Bytes
  KB,  ///< Kilobytes (1024 Bytes)
  MB,  ///< Megabytes (1024 Kilobytes)
  GB   ///< Gigabytes (1024 Megabytes)
};

/**
 * Get the current Resident Set Size (RSS) in bytes.
 * This is the portion of memory occupied by a process that is held in RAM.
 * @note This is different from the virtual memory size, which includes all memory the process can access,
 * including memory that is swapped out to disk or allocated but not used.
 * @return current RSS in bytes
 */
std::size_t get_current_rss();
/**
 * Get the peak Resident Set Size (RSS) in bytes.
 * This is the maximum portion of memory occupied by a process that was held in RAM at any point in time.
 * @note This is different from the virtual memory size, which includes all memory the process can access,
 * including memory that is swapped out to disk or allocated but not used.
 * @return peak RSS in bytes
 */
std::size_t get_peak_rss();

/**
 * Convert a size in bytes to the specified memory unit.
 * @tparam U memory unit to convert to
 * @param size_in_bytes size in bytes
 * @return size in the specified memory unit
 */
template <MemoryUnit U>
double bytes_to(const std::size_t size_in_bytes) {
  if constexpr (U == MemoryUnit::B) return static_cast<double>(size_in_bytes);
  if constexpr (U == MemoryUnit::KB) return static_cast<double>(size_in_bytes) / static_cast<double>(1 << 10);
  if constexpr (U == MemoryUnit::MB) return static_cast<double>(size_in_bytes) / static_cast<double>(1 << 20);
  if constexpr (U == MemoryUnit::GB) return static_cast<double>(size_in_bytes) / static_cast<double>(1 << 30);
  return 0.0;
}

/**
 * Convert a size in bytes to the specified memory unit.
 * @param unit memory unit to convert to
 * @param size_in_bytes size in bytes
 * @return size in the specified memory unit
 */
double bytes_to(std::size_t size_in_bytes, MemoryUnit unit);

/**
 * Suggest the most appropriate memory unit for a given size in bytes.
 * The function selects the largest memory unit such that the converted size is at least 1 and less than 1024,
 * provided the size is non-zero and within the range of known memory units.
 * @code
 * get_suggested_memory_unit(500);            // Returns MemoryUnit::B  -> 500 B
 * get_suggested_memory_unit(2048);           // Returns MemoryUnit::KB -> 2 KB
 * get_suggested_memory_unit(5'242'880);      // Returns MemoryUnit::MB -> 5 MB
 * get_suggested_memory_unit(10'737'418'240); // Returns MemoryUnit::GB -> 10 GB
 * @endcode
 * @param size_in_bytes size in bytes
 * @return suggested memory unit for the given size in bytes
 */
MemoryUnit get_suggested_memory_unit(std::size_t size_in_bytes);

std::ostream& operator<<(std::ostream& os, MemoryUnit unit);

}  // namespace lucid::metrics

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::metrics::MemoryUnit)

#endif
