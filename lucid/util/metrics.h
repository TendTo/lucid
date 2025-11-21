/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Collection of metric measuring functions.
 */
#pragma once

#include <chrono>
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

/** Time units for conversion. */
enum class TimeUnit {
  MS,  ///< Milliseconds
  S,   ///< Seconds
  M,   ///< Minutes (60 Seconds)
  H,   ///< Hours (60 Minutes)
  D    ///< Days (24 Hours)
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
 * provided the size is in the range of known memory units.
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

/**
 * Convert a duration in seconds to the specified time unit.
 * @tparam U time unit to convert to
 * @param duration_in_seconds duration in seconds
 * @return duration in the specified time unit
 */
template <TimeUnit U>
double time_to(const double duration_in_seconds) {
  if constexpr (U == TimeUnit::MS) return duration_in_seconds * 1e3;
  if constexpr (U == TimeUnit::S) return duration_in_seconds;
  if constexpr (U == TimeUnit::M) return duration_in_seconds / 60.0;
  if constexpr (U == TimeUnit::H) return duration_in_seconds / 3600.0;
  if constexpr (U == TimeUnit::D) return duration_in_seconds / 86400.0;
  return 0.0;
}

/**
 * Convert a duration in seconds to the specified time unit.
 * @param duration_in_seconds duration in seconds
 * @param unit time unit to convert to
 * @return duration in the specified time unit
 */
double time_to(double duration_in_seconds, TimeUnit unit);

/**
 * Suggest the most appropriate time unit for a given duration in seconds.
 * The function selects the largest time unit such that the converted duration is around 1
 * and less than the next larger unit,
 * provided the duration in the range of known time units.
 * @code
 * get_suggested_time_unit(0.002);        // Returns TimeUnit::MS  -> 2 ms
 * get_suggested_time_unit(0.5);          // Returns TimeUnit::S   -> 0.5 s
 * get_suggested_time_unit(2);            // Returns TimeUnit::S   -> 2 s
 * get_suggested_time_unit(120);          // Returns TimeUnit::M   -> 2 min
 * get_suggested_time_unit(7200);         // Returns TimeUnit::H   -> 2 h
 * get_suggested_time_unit(172800);      // Returns TimeUnit::D    -> 2 d
 * @endcode
 * @note Seconds has a larger range to capture ranges over a 1/100th of a second up to a minute.
 * @param duration_in_seconds duration in seconds
 * @return suggested time unit for the given duration in seconds
 */
TimeUnit get_suggested_time_unit(double duration_in_seconds);

std::ostream& operator<<(std::ostream& os, MemoryUnit unit);

}  // namespace lucid::metrics

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::metrics::MemoryUnit)

#endif
