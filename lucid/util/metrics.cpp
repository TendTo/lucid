/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Adapted from https://maxliani.wordpress.com/2020/05/02/dev-tracking-memory-usage-part-1/
 */
#include "lucid/util/metrics.h"

#include <cstddef>

#include "lucid/util/error.h"

namespace lucid::metrics {
#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>

#include <cstdio>

std::size_t get_current_rss() {
  // The value we query later on is measured in number of pages. Query the size
  // of a page in bytes. This is typically 4KB, but pages could be also configured
  // at the system level to be 2MB.
  static size_t page_size = sysconf(_SC_PAGESIZE);

  FILE* const stat_file = fopen("/proc/self/statm", "r");
  if (!stat_file) return 0;

  // Attempt to read the value we need, it this won't succeed, size will be left
  // at zero.
  std::size_t _, pages_count = 0;
  [[maybe_unused]] const int res = fscanf(stat_file, "%zu %zu", &_, &pages_count);
  LUCID_ASSERT(res == 2, "Failed to read memory usage from /proc/self/statm");
  fclose(stat_file);  // NOLINT(build/include_what_you_use): Does not detect cstdio

  // Compute the size in bytes.
  return pages_count * page_size;
}

std::size_t get_peak_rss() {
  rusage usage{};
  if (0 != getrusage(RUSAGE_SELF, &usage)) LUCID_RUNTIME_ERROR("Failed to get current resource usage (getrusage)");

  // From "man getrusage":
  // ru_maxrss (since Linux 2.6.32)
  //           This is the maximum resident set size used (in kilobytes).
  return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
}
#elif defined(_WIN32)
#include <windows.h>
// Include order matters, and 'windows.h' must be included before 'psapi.h'.
#include <psapi.h>

size_t get_current_rss() {
  // Obtain a handle to the current process, which is what we want to measure.
  // The process handle is not going to change, so we only do it once.
  static HANDLE process = GetCurrentProcess();

  PROCESS_MEMORY_COUNTERS counters;
  GetProcessMemoryInfo(process, &counters, sizeof(PROCESS_MEMORY_COUNTERS));
  return counters.WorkingSetSize;
}

size_t get_peak_rss() {
  // Obtain a handle to the current process, which is what we want to measure.
  // The process handle is not going to change, so we only do it once.
  static HANDLE process = GetCurrentProcess();

  PROCESS_MEMORY_COUNTERS counters;
  GetProcessMemoryInfo(process, &counters, sizeof(PROCESS_MEMORY_COUNTERS));
  return counters.PeakWorkingSetSize;
}
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#include <unistd.h>

size_t get_current_rss() {
  // Configure what we want to query
  mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

  // Retrieve basic information about the process
  kern_return_t result = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);
  if (result != KERN_SUCCESS) return 0;

  return size_t(info.resident_size);
}

size_t get_peak_rss() {
  rusage usage_data;
  getrusage(RUSAGE_SELF, &usage_data);

  // From "man getrusage":
  //    ru_maxrss the maximum resident set size utilized (in bytes).
  return size_t(usage_data.ru_maxrss);
}
#else
size_t get_current_rss() { return 0; }
size_t get_peak_rss() { return 0; }
#endif

double bytes_to(const std::size_t size_in_bytes, const MemoryUnit unit) {
  switch (unit) {
    case MemoryUnit::B:
      return static_cast<double>(size_in_bytes);
    case MemoryUnit::KB:
      return static_cast<double>(size_in_bytes) / static_cast<double>(1 << 10);
    case MemoryUnit::MB:
      return static_cast<double>(size_in_bytes) / static_cast<double>(1 << 20);
    case MemoryUnit::GB:
      return static_cast<double>(size_in_bytes) / static_cast<double>(1 << 30);
    default:
      return 0.0;
  }
}

MemoryUnit get_suggested_memory_unit(const std::size_t size_in_bytes) {
  if (size_in_bytes < (1 << 10)) return MemoryUnit::B;
  if (size_in_bytes < (1 << 20)) return MemoryUnit::KB;
  if (size_in_bytes < (1 << 30)) return MemoryUnit::MB;
  return MemoryUnit::GB;
}

std::ostream& operator<<(std::ostream& os, MemoryUnit unit) {
  switch (unit) {
    case MemoryUnit::B:
      return os << "B";
    case MemoryUnit::KB:
      return os << "KB";
    case MemoryUnit::MB:
      return os << "MB";
    case MemoryUnit::GB:
      return os << "GB";
    default:
      LUCID_UNREACHABLE();
  }
}

}  // namespace lucid::metrics
