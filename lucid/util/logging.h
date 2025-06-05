/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Logging macros.
 * Allows logging with different verbosity levels using spdlog.
 * The verbosity level is set with the -V flag.
 * The verbosity level is an integer between 0 and 5 and it increases with each -V flag.
 * It can be reduced with the -q flag.
 * It starts at 2 (warning).
 */
#pragma once

#include <fmt/core.h>     // IWYU pragma: export
#include <fmt/ostream.h>  // IWYU pragma: export
#include <fmt/ranges.h>   // IWYU pragma: export

#define OSTREAM_FORMATTER(type) \
  template <>                   \
  struct fmt::formatter<type> : ostream_formatter {};

#define LUCID_FORMAT_MATRIX_SHAPE(matrix) fmt::format("[{}x{}]", (matrix).rows(), (matrix).cols())

#ifndef NLOG

#include <spdlog/logger.h>

#include <memory>

namespace lucid {

enum class LoggerType { OUT, ERR };

std::shared_ptr<spdlog::logger> get_logger(LoggerType logger_type);

}  // namespace lucid

#ifdef _MSC_VER
#define LUCID_FUNCTION_SIGNATURE __FUNCTION__  // or __FUNCSIG__
#else
consteval std::string_view method_name(const char *s) {
  const std::string_view prettyFunction(s);
  const std::size_t bracket = prettyFunction.rfind('(');
  const std::size_t space = prettyFunction.rfind(' ', bracket) + 1;
  return prettyFunction.substr(space, bracket - space);
}
#define LUCID_FUNCTION_SIGNATURE method_name(__PRETTY_FUNCTION__)
#endif

#define LUCID_FORMAT(message, ...) fmt::format(message, __VA_ARGS__)

#define LUCID_VERBOSITY_TO_LOG_LEVEL(verbosity)                        \
  ((verbosity) == 0                                                    \
       ? spdlog::level::critical                                       \
       : ((verbosity) == 1                                             \
              ? spdlog::level::err                                     \
              : ((verbosity) == 2                                      \
                     ? spdlog::level::warn                             \
                     : ((verbosity) == 3                               \
                            ? spdlog::level::info                      \
                            : ((verbosity) == 4 ? spdlog::level::debug \
                                                : ((verbosity) == 5 ? spdlog::level::trace : spdlog::level::off))))))
#define LUCID_LOG_INIT_VERBOSITY(verbosity) LUCID_LOG_INIT_LEVEL(LUCID_VERBOSITY_TO_LOG_LEVEL(verbosity))
#define LUCID_LOG_INIT_LEVEL(level)                                  \
  do {                                                               \
    ::lucid::get_logger(::lucid::LoggerType::OUT)->set_level(level); \
    ::lucid::get_logger(::lucid::LoggerType::ERR)->set_level(level); \
  } while (0)
#define LUCID_LOG_MSG(msg) "[{}] " msg, LUCID_FUNCTION_SIGNATURE
#define LUCID_TRACE(msg) ::lucid::get_logger(::lucid::LoggerType::OUT)->trace(msg)
#define LUCID_TRACE_FMT(msg, ...) ::lucid::get_logger(::lucid::LoggerType::OUT)->trace(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_DEBUG(msg) ::lucid::get_logger(::lucid::LoggerType::OUT)->debug(LUCID_LOG_MSG(msg))
#define LUCID_DEBUG_FMT(msg, ...) ::lucid::get_logger(::lucid::LoggerType::OUT)->debug(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_INFO(msg) ::lucid::get_logger(::lucid::LoggerType::OUT)->info(LUCID_LOG_MSG(msg))
#define LUCID_INFO_FMT(msg, ...) ::lucid::get_logger(::lucid::LoggerType::OUT)->info(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_WARN(msg) ::lucid::get_logger(::lucid::LoggerType::ERR)->warn(LUCID_LOG_MSG(msg))
#define LUCID_WARN_FMT(msg, ...) ::lucid::get_logger(::lucid::LoggerType::ERR)->warn(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_ERROR(msg) ::lucid::get_logger(::lucid::LoggerType::ERR)->error(LUCID_LOG_MSG(msg))
#define LUCID_ERROR_FMT(msg, ...) ::lucid::get_logger(::lucid::LoggerType::ERR)->error(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_CRITICAL(msg) ::lucid::get_logger(::lucid::LoggerType::ERR)->critical(LUCID_LOG_MSG(msg))
#define LUCID_CRITICAL_FMT(msg, ...) \
  ::lucid::get_logger(::lucid::LoggerType::ERR)->critical(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_INFO_ENABLED (::lucid::get_logger(::lucid::LoggerType::OUT)->should_log(spdlog::level::info))
#define LUCID_DEBUG_ENABLED (::lucid::get_logger(::lucid::LoggerType::OUT)->should_log(spdlog::level::debug))
#define LUCID_TRACE_ENABLED (::lucid::get_logger(::lucid::LoggerType::OUT)->should_log(spdlog::level::trace))

#ifndef NDEBUG

#include <iostream>
#include <thread>

#define LUCID_DEV(msg)                                                                                          \
  do {                                                                                                          \
    if (::lucid::get_logger(::lucid::LoggerType::ERR)->should_log(spdlog::level::err))                          \
      fmt::println("[{:%Y-%m-%d %H:%M:%S}] [\033[1m\033[35mDEV\033[0m] [thread {}] " msg "",                    \
                   std::chrono::system_clock::now(), std::hash<std::thread::id>{}(std::this_thread::get_id())); \
  } while (0)
#define LUCID_DEV_FMT(msg, ...)                                                                                \
  do {                                                                                                         \
    if (::lucid::get_logger(::lucid::LoggerType::ERR)->should_log(spdlog::level::err))                         \
      fmt::println("[{:%Y-%m-%d %H:%M:%S}] [\033[1m\033[35mDEV\033[0m] [thread {}] " msg "",                   \
                   std::chrono::system_clock::now(), std::hash<std::thread::id>{}(std::this_thread::get_id()), \
                   __VA_ARGS__);                                                                               \
  } while (0)

#define LUCID_DEV_TRACE(msg) LUCID_DEV(msg)
#define LUCID_DEV_TRACE_FMT(msg, ...) LUCID_DEV_FMT(msg, __VA_ARGS__)
#define LUCID_DEV_DEBUG(msg) LUCID_DEV(msg)
#define LUCID_DEV_DEBUG_FMT(msg, ...) LUCID_DEV_FMT(msg, __VA_ARGS__)
#else
#define LUCID_DEV(msg) void(0)
#define LUCID_DEV_FMT(msg, ...) void(0)
#define LUCID_DEV_TRACE(msg) LUCID_TRACE(msg)
#define LUCID_DEV_TRACE_FMT(msg, ...) LUCID_TRACE_FMT(msg, __VA_ARGS__)
#define LUCID_DEV_DEBUG(msg) LUCID_DEBUG(msg)
#define LUCID_DEV_DEBUG_FMT(msg, ...) LUCID_DEBUG_FMT(msg, __VA_ARGS__)
#endif

#else

#define LUCID_FORMAT(message, ...) fmt::format(message, __VA_ARGS__)
#define LUCID_VERBOSITY_TO_LOG_LEVEL(verbosity) 0
#define LUCID_LOG_INIT_LEVEL(level) void(0)
#define LUCID_LOG_INIT_VERBOSITY(verbosity) void(0)
#define LUCID_TRACE(msg) void(0)
#define LUCID_TRACE_FMT(msg, ...) void(0)
#define LUCID_DEBUG(msg) void(0)
#define LUCID_DEBUG_FMT(msg, ...) void(0)
#define LUCID_INFO(msg) void(0)
#define LUCID_INFO_FMT(msg, ...) void(0)
#define LUCID_WARN(msg) void(0)
#define LUCID_WARN_FMT(msg, ...) void(0)
#define LUCID_ERROR(msg) void(0)
#define LUCID_ERROR_FMT(msg, ...) void(0)
#define LUCID_CRITICAL(msg) void(0)
#define LUCID_CRITICAL_FMT(msg, ...) void(0)
#define LUCID_INFO_ENABLED false
#define LUCID_DEBUG_ENABLED false
#define LUCID_TRACE_ENABLED false
#define LUCID_DEV(msg) void(0)
#define LUCID_DEV_FMT(msg, ...) void(0)
#define LUCID_DEV_TRACE(msg) void(0)
#define LUCID_DEV_TRACE_FMT(msg, ...) void(0)
#define LUCID_DEV_DEBUG(msg) void(0)
#define LUCID_DEV_DEBUG_FMT(msg, ...) void(0)

#endif
