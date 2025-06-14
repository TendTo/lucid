/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Logging macros.
 * Allows logging with different verbosity levels using spdlog.
 * The verbosity level is an integer between 0 and 5.
 * If set to -1, logging is disabled.
 * It starts at 3 (info).
 */
#pragma once

#include <fmt/core.h>     // IWYU pragma: export
#include <fmt/ostream.h>  // IWYU pragma: export
#include <fmt/ranges.h>   // IWYU pragma: export

constexpr int LUCID_LOG_OFF_LEVEL = -1;
constexpr int LUCID_LOG_CRITICAL_LEVEL = 0;
constexpr int LUCID_LOG_ERROR_LEVEL = 1;
constexpr int LUCID_LOG_WARN_LEVEL = 2;
constexpr int LUCID_LOG_INFO_LEVEL = 3;
constexpr int LUCID_LOG_DEBUG_LEVEL = 4;
constexpr int LUCID_LOG_TRACE_LEVEL = 5;

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
consteval std::string_view function_signature(const char *s) {
  const std::string_view prompt{s};
  return prompt.starts_with("init_util") ? std::string_view{"-"} : prompt;
}
#define LUCID_FUNCTION_SIGNATURE function_signature(__FUNCTION__)
#else
consteval std::string_view function_signature(const char *s) {
  const std::string_view prettyFunction{s};
  const std::size_t bracket = prettyFunction.rfind('(');
  const std::size_t space = prettyFunction.rfind(' ', bracket) + 1;
  const std::string_view prompt{prettyFunction.substr(space, bracket - space)};
  return prompt.starts_with("init_util") ? std::string_view{"-"} : prompt;
}
#define LUCID_FUNCTION_SIGNATURE function_signature(__PRETTY_FUNCTION__)
#endif

#define LUCID_FORMAT(message, ...) fmt::format(message, __VA_ARGS__)

#define LUCID_VERBOSITY_TO_LOG_LEVEL(verbosity)                                                   \
  ((verbosity) == LUCID_LOG_CRITICAL_LEVEL                                                        \
       ? spdlog::level::critical                                                                  \
       : ((verbosity) == LUCID_LOG_ERROR_LEVEL                                                    \
              ? spdlog::level::err                                                                \
              : ((verbosity) == LUCID_LOG_WARN_LEVEL                                              \
                     ? spdlog::level::warn                                                        \
                     : ((verbosity) == LUCID_LOG_INFO_LEVEL                                       \
                            ? spdlog::level::info                                                 \
                            : ((verbosity) == LUCID_LOG_DEBUG_LEVEL                               \
                                   ? spdlog::level::debug                                         \
                                   : ((verbosity) == LUCID_LOG_TRACE_LEVEL ? spdlog::level::trace \
                                                                           : spdlog::level::off))))))
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

#endif
