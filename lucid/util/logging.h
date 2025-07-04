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

#ifdef LUCID_VERBOSE_EIGEN_BUILD
#define LUCID_FORMAT_MATRIX(matrix) fmt::format("[{}x{}]\n{}", (matrix).rows(), (matrix).cols(), matrix)
#else
#define LUCID_FORMAT_MATRIX(matrix) fmt::format("[{}x{}]", (matrix).rows(), (matrix).cols())
#endif

#ifndef NLOG

#include <spdlog/logger.h>
#include <spdlog/sinks/callback_sink.h>

#include <memory>

/**
 * @namespace lucid::log
 * Collection of logging utilities.
 */
namespace lucid::log {

/** Enum used to differentiate between loggers that output to stdout and stderr. */
enum class LoggerType { OUT, ERR };

/**
 * Get a logger of the specified type.
 * If the logger does not exist, it will be created.
 * @param logger_type stdout or stderr logger
 * @return shared pointer to the logger
 */
std::shared_ptr<spdlog::logger> get_logger(LoggerType logger_type);
/**
 * Set the verbosity level of the logger.
 * @param level verbosity level to set
 */
void set_verbosity_level(spdlog::level::level_enum level);
/**
 * Replace the standard logger sink with a custom callback.
 * This action can be undone by calling @ref clear_logger.
 * @note The logger will no longer output to stdout or stderr, unless the callback does so.
 * @param cb custom callback that will be called with the log message
 */
void set_logger_sink(spdlog::custom_log_callback cb);
/**
 * Replace the standard logger sink with a custom callback.
 * This action can be undone by calling @ref clear_logger.
 * @note The logger will no longer output to stdout or stderr.
 * @param cb custom callback that will be called with the log message
 */
void set_logger_sink(std::function<void(std::string)> cb);
/**
 * Clear the logger, removing all loggers and their sinks.
 * This will remove all loggers created by @ref get_logger and @ref set_logger_sink.
 * After this call, the loggers will need to be recreated using @ref get_logger.
 */
void clear_logger();

}  // namespace lucid::log

#ifdef _MSC_VER
consteval std::string_view function_signature(const char* s) {
  const std::string_view prompt{s};
  return prompt.starts_with("init_util") ? std::string_view{"-"} : prompt;
}
#define LUCID_FUNCTION_SIGNATURE function_signature(__FUNCTION__)
#else
consteval std::string_view function_signature(const char* s) {
  const std::string_view prettyFunction{s};
  const std::size_t bracket = prettyFunction.rfind('(');
  const std::size_t space = prettyFunction.rfind(' ', bracket) + 1;
  const std::string_view prompt{prettyFunction.substr(space, bracket - space)};
  return prompt.starts_with("init_util") ? std::string_view{"-"} : prompt;
}
#define LUCID_FUNCTION_SIGNATURE function_signature(__PRETTY_FUNCTION__)
#endif

#define LUCID_FORMAT(message, ...) fmt::format(message, __VA_ARGS__)
#define LUCID_OUT_LOGGER ::lucid::log::get_logger(lucid::log::LoggerType::OUT)
#define LUCID_ERR_LOGGER ::lucid::log::get_logger(lucid::log::LoggerType::ERR)

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
#define LUCID_LOG_INIT_VERBOSITY(verbosity) ::lucid::log::set_verbosity_level(LUCID_VERBOSITY_TO_LOG_LEVEL(verbosity))
#define LUCID_LOG_MSG(msg) "[{}] " msg, LUCID_FUNCTION_SIGNATURE
#define LUCID_TRACE(msg) LUCID_OUT_LOGGER->trace(LUCID_LOG_MSG(msg))
#define LUCID_TRACE_FMT(msg, ...) LUCID_OUT_LOGGER->trace(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_DEBUG(msg) LUCID_OUT_LOGGER->debug(LUCID_LOG_MSG(msg))
#define LUCID_DEBUG_FMT(msg, ...) LUCID_OUT_LOGGER->debug(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_INFO(msg) LUCID_OUT_LOGGER->info(LUCID_LOG_MSG(msg))
#define LUCID_INFO_FMT(msg, ...) LUCID_OUT_LOGGER->info(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_WARN(msg) LUCID_ERR_LOGGER->warn(LUCID_LOG_MSG(msg))
#define LUCID_WARN_FMT(msg, ...) LUCID_ERR_LOGGER->warn(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_ERROR(msg) LUCID_ERR_LOGGER->error(LUCID_LOG_MSG(msg))
#define LUCID_ERROR_FMT(msg, ...) LUCID_ERR_LOGGER->error(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_CRITICAL(msg) LUCID_ERR_LOGGER->critical(LUCID_LOG_MSG(msg))
#define LUCID_CRITICAL_FMT(msg, ...) LUCID_ERR_LOGGER->critical(LUCID_LOG_MSG(msg), __VA_ARGS__)
#define LUCID_TRACE_ENABLED (LUCID_OUT_LOGGER->should_log(spdlog::level::trace))
#define LUCID_DEBUG_ENABLED (LUCID_OUT_LOGGER->should_log(spdlog::level::debug))
#define LUCID_INFO_ENABLED (LUCID_OUT_LOGGER->should_log(spdlog::level::info))
#define LUCID_WARN_ENABLED (LUCID_ERR_LOGGER->should_log(spdlog::level::warn))
#define LUCID_ERROR_ENABLED (LUCID_ERR_LOGGER->should_log(spdlog::level::err))
#define LUCID_CRITICAL_ENABLED (LUCID_ERR_LOGGER->should_log(spdlog::level::critical))

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
#define LUCID_TRACE_ENABLED false
#define LUCID_DEBUG_ENABLED false
#define LUCID_INFO_ENABLED false
#define LUCID_WARN_ENABLED false
#define LUCID_ERROR_ENABLED false
#define LUCID_CRITICAL_ENABLED false

#endif
