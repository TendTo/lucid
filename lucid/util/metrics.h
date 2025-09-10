/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Collection of metric measuring functions.
 */
#pragma once

#include <cstddef>

/** Memory units for conversion. */
enum class MemoryUnit {
  B,   ///< Bytes
  KB,  ///< Kilobytes (1024 Bytes)
  MB,  ///< Megabytes (1024 Kilobytes)
  GB   ///< Gigabytes (1024 Megabytes)
};

/**
 * @namespace lucid::metrics
 * Collection of metric measuring functions.
 */
namespace lucid::metrics {

/**
 * Get the current Resident Set Size (RSS) in bytes.
 * This is the portion of memory occupied by a process that is held in RAM.
 * @return current RSS in bytes
 */
std::size_t get_current_rss();
/**
 * Get the peak Resident Set Size (RSS) in bytes.
 * This is the maximum portion of memory occupied by a process that was held in RAM at any point in time.
 * @return peak RSS in bytes
 */
std::size_t get_peak_rss();

/**
 * Convert a size in bytes to the specified memory unit.
 * @tparam U memory unit to convert to (default: MB)
 * @param size_in_bytes size in bytes
 * @return size in megabytes
 */
template <MemoryUnit U = MemoryUnit::MB>
double bytes_to(const std::size_t size_in_bytes) {
  if constexpr (U == MemoryUnit::B) return static_cast<double>(size_in_bytes);
  if constexpr (U == MemoryUnit::KB) return static_cast<double>(size_in_bytes) / static_cast<double>(1 << 10);
  if constexpr (U == MemoryUnit::MB) return static_cast<double>(size_in_bytes) / static_cast<double>(1 << 20);
  if constexpr (U == MemoryUnit::GB) return static_cast<double>(size_in_bytes) / static_cast<double>(1 << 30);
  return 0.0;
}

}  // namespace lucid::metrics
