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

constexpr int LUCID_LOG_OFF_LEVEL = -1;      ///< Verbosity level to disable all logging.
constexpr int LUCID_LOG_CRITICAL_LEVEL = 0;  ///< Verbosity level for critical messages only.
constexpr int LUCID_LOG_ERROR_LEVEL = 1;     ///< Verbosity level for error and critical messages.
constexpr int LUCID_LOG_WARN_LEVEL = 2;      ///< Verbosity level for warning, error and critical messages.
constexpr int LUCID_LOG_INFO_LEVEL = 3;      ///< Verbosity level for info, warning, error and critical messages.
constexpr int LUCID_LOG_DEBUG_LEVEL = 4;     ///< Verbosity level for debug, info, warning, error and critical messages.
constexpr int LUCID_LOG_TRACE_LEVEL = 5;     ///< Verbosity level for all messages including trace.

/**
 * Define a fmt formatter for a type using ostream formatting.
 * This macro creates a formatter specialization that uses the type's stream operator.
 * @param type the type to create a formatter for
 */
#define OSTREAM_FORMATTER(type) \
  template <>                   \
  struct fmt::formatter<type> : ostream_formatter {};

/**
 * Format a message using fmt::format with variadic arguments (NLOG version).
 * When NLOG is defined, this still provides formatting capability.
 * @param message format string
 * @param ... arguments to format into the message
 * @return formatted string
 */
#define LUCID_FORMAT(message, ...) fmt::format(message, __VA_ARGS__)

#ifdef LUCID_VERBOSE_EIGEN_BUILD
/**
 * Format an Eigen vector with verbose output including its contents.
 * @param vector the Eigen vector to format
 * @return formatted string with dimensions and full vector contents
 */
#define LUCID_FORMAT_VECTOR(vector) fmt::format("[1x{}]\n{}", (vector).size(), vector)
/**
 * Format an Eigen matrix with verbose output including its contents.
 * @param matrix the Eigen matrix to format
 * @return formatted string with dimensions and full matrix contents
 */
#define LUCID_FORMAT_MATRIX(matrix) fmt::format("[{}x{}]\n{}", (matrix).rows(), (matrix).cols(), matrix)
#else
/**
 * Format an Eigen vector with compact output showing only dimensions.
 * @param vector the Eigen vector to format
 * @return formatted string with dimensions only
 */
#define LUCID_FORMAT_VECTOR(vector) fmt::format("[1x{}]", (vector).size())
/**
 * Format an Eigen matrix with compact output showing only dimensions.
 * @param matrix the Eigen matrix to format
 * @return formatted string with dimensions only
 */
#define LUCID_FORMAT_MATRIX(matrix) fmt::format("[{}x{}]", (matrix).rows(), (matrix).cols())
#endif

#ifndef NLOG

#include <spdlog/logger.h>
#include <spdlog/sinks/callback_sink.h>

#include <memory>
#include <string>

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
 * Set the verbosity level of the logger.
 * @param level verbosity level to set
 */
void set_verbosity_level(int level);
/**
 * Replace the standard logger sink with a custom callback.
 * This action can be undone by calling @ref clear_logger.
 * @note The logger will no longer output to stdout or stderr, unless `cb` explicitly does so.
 * @param cb custom callback that will be called with the log message
 */
void set_logger_sink(spdlog::custom_log_callback cb);
/**
 * Replace the standard logger sink with a custom callback.
 * This action can be undone by calling @ref clear_logger.
 * @note The logger will no longer output to stdout or stderr, unless `cb` explicitly does so.
 * @param cb custom callback that will be called with the log message
 */
void set_logger_sink(std::function<void(std::string)> cb);
/**
 * Set the format of the logger.
 * This will replace the default format `[%%Y-%%m-%%d %%H:%%M:%%S.%%e] [%^%l%$] [thread %%t] %%v`.
 * @note See [spdlog's documentation](https://github.com/gabime/spdlog/wiki/Custom-formatting)
 * for more information on the format string.
 * @param format format string to use for the logger
 */
void set_pattern(const std::string& format);
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
/**
 * Extract function name from compiler-provided function signature on MSVC.
 * Uses __FUNCTION__ which provides simpler function names on MSVC.
 */
#define LUCID_FUNCTION_SIGNATURE function_signature(__FUNCTION__)
#else
consteval std::string_view function_signature(const char* s) {
  const std::string_view prettyFunction{s};
  const std::size_t bracket = prettyFunction.rfind('(');
  const std::size_t space = prettyFunction.rfind(' ', bracket) + 1;
  const std::string_view prompt{prettyFunction.substr(space, bracket - space)};
  return prompt.starts_with("init_util") ? std::string_view{"-"} : prompt;
}
/**
 * Extract function name from compiler-provided function signature on GCC/Clang.
 * Uses __PRETTY_FUNCTION__ which provides more detailed function signatures.
 */
#define LUCID_FUNCTION_SIGNATURE function_signature(__PRETTY_FUNCTION__)
#endif

/**
 * Get the stdout logger instance.
 * @return shared pointer to the stdout logger
 */
#define LUCID_OUT_LOGGER ::lucid::log::get_logger(lucid::log::LoggerType::OUT)
/**
 * Get the stderr logger instance.
 * @return shared pointer to the stderr logger
 */
#define LUCID_ERR_LOGGER ::lucid::log::get_logger(lucid::log::LoggerType::ERR)

/**
 * Convert a lucid verbosity level to the corresponding spdlog level.
 * @param verbosity the lucid verbosity level (0-5, or -1 for off)
 * @return corresponding spdlog::level::level_enum value
 */
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
/**
 * Initialize the logger with the specified verbosity level.
 * @param verbosity the verbosity level to set (0-5, or -1 for off)
 */
#define LUCID_LOG_INIT_VERBOSITY(verbosity) ::lucid::log::set_verbosity_level(LUCID_VERBOSITY_TO_LOG_LEVEL(verbosity))
/**
 * Format a log message with function signature prefix.
 * @param msg the message to format
 * @return formatted message string with function signature prefix
 */
#define LUCID_LOG_MSG(msg) "[{}] " msg, LUCID_FUNCTION_SIGNATURE
/**
 * Log a trace level message.
 * @param msg the message to log
 */
#define LUCID_TRACE(msg) LUCID_OUT_LOGGER->trace(LUCID_LOG_MSG(msg))
/**
 * Log a trace level message with formatted arguments.
 * @param msg the format string
 * @param ... arguments to format into the message
 */
#define LUCID_TRACE_FMT(msg, ...) LUCID_OUT_LOGGER->trace(LUCID_LOG_MSG(msg), __VA_ARGS__)
/**
 * Log a debug level message.
 * @param msg the message to log
 */
#define LUCID_DEBUG(msg) LUCID_OUT_LOGGER->debug(LUCID_LOG_MSG(msg))
/**
 * Log a debug level message with formatted arguments.
 * @param msg the format string
 * @param ... arguments to format into the message
 */
#define LUCID_DEBUG_FMT(msg, ...) LUCID_OUT_LOGGER->debug(LUCID_LOG_MSG(msg), __VA_ARGS__)
/**
 * Log an info level message.
 * @param msg the message to log
 */
#define LUCID_INFO(msg) LUCID_OUT_LOGGER->info(LUCID_LOG_MSG(msg))
/**
 * Log an info level message with formatted arguments.
 * @param msg the format string
 * @param ... arguments to format into the message
 */
#define LUCID_INFO_FMT(msg, ...) LUCID_OUT_LOGGER->info(LUCID_LOG_MSG(msg), __VA_ARGS__)
/**
 * Log a warning level message.
 * @param msg the message to log
 */
#define LUCID_WARN(msg) LUCID_ERR_LOGGER->warn(LUCID_LOG_MSG(msg))
/**
 * Log a warning level message with formatted arguments.
 * @param msg the format string
 * @param ... arguments to format into the message
 */
#define LUCID_WARN_FMT(msg, ...) LUCID_ERR_LOGGER->warn(LUCID_LOG_MSG(msg), __VA_ARGS__)
/**
 * Log an error level message.
 * @param msg the message to log
 */
#define LUCID_ERROR(msg) LUCID_ERR_LOGGER->error(LUCID_LOG_MSG(msg))
/**
 * Log an error level message with formatted arguments.
 * @param msg the format string
 * @param ... arguments to format into the message
 */
#define LUCID_ERROR_FMT(msg, ...) LUCID_ERR_LOGGER->error(LUCID_LOG_MSG(msg), __VA_ARGS__)
/**
 * Log a critical level message.
 * @param msg the message to log
 */
#define LUCID_CRITICAL(msg) LUCID_ERR_LOGGER->critical(LUCID_LOG_MSG(msg))
/**
 * Log a critical level message with formatted arguments.
 * @param msg the format string
 * @param ... arguments to format into the message
 */
#define LUCID_CRITICAL_FMT(msg, ...) LUCID_ERR_LOGGER->critical(LUCID_LOG_MSG(msg), __VA_ARGS__)
/**
 * Check if trace level logging is enabled.
 * @return true if trace logging is enabled, false otherwise
 */
#define LUCID_TRACE_ENABLED (LUCID_OUT_LOGGER->should_log(spdlog::level::trace))
/**
 * Check if debug level logging is enabled.
 * @return true if debug logging is enabled, false otherwise
 */
#define LUCID_DEBUG_ENABLED (LUCID_OUT_LOGGER->should_log(spdlog::level::debug))
/**
 * Check if info level logging is enabled.
 * @return true if info logging is enabled, false otherwise
 */
#define LUCID_INFO_ENABLED (LUCID_OUT_LOGGER->should_log(spdlog::level::info))
/**
 * Check if warning level logging is enabled.
 * @return true if warning logging is enabled, false otherwise
 */
#define LUCID_WARN_ENABLED (LUCID_ERR_LOGGER->should_log(spdlog::level::warn))
/**
 * Check if error level logging is enabled.
 * @return true if error logging is enabled, false otherwise
 */
#define LUCID_ERROR_ENABLED (LUCID_ERR_LOGGER->should_log(spdlog::level::err))
/**
 * Check if critical level logging is enabled.
 * @return true if critical logging is enabled, false otherwise
 */
#define LUCID_CRITICAL_ENABLED (LUCID_ERR_LOGGER->should_log(spdlog::level::critical))

#else

/**
 * Convert a lucid verbosity level to spdlog level (NLOG version).
 * When NLOG is defined, this always returns 0 since logging is disabled.
 * @param verbosity the verbosity level (ignored)
 * @return always returns 0
 */
#define LUCID_VERBOSITY_TO_LOG_LEVEL(verbosity) 0
/**
 * Initialize logger with a specific level (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param level the level to set (ignored)
 */
#define LUCID_LOG_INIT_LEVEL(level) void(0)
/**
 * Initialize logger with verbosity level (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param verbosity the verbosity level (ignored)
 */
#define LUCID_LOG_INIT_VERBOSITY(verbosity) void(0)
/**
 * Log a trace message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the message (ignored)
 */
#define LUCID_TRACE(msg) void(0)
/**
 * Log a formatted trace message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the format string (ignored)
 * @param ... format arguments (ignored)
 */
#define LUCID_TRACE_FMT(msg, ...) void(0)
/**
 * Log a debug message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the message (ignored)
 */
#define LUCID_DEBUG(msg) void(0)
/**
 * Log a formatted debug message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the format string (ignored)
 * @param ... format arguments (ignored)
 */
#define LUCID_DEBUG_FMT(msg, ...) void(0)
/**
 * Log an info message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the message (ignored)
 */
#define LUCID_INFO(msg) void(0)
/**
 * Log a formatted info message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the format string (ignored)
 * @param ... format arguments (ignored)
 */
#define LUCID_INFO_FMT(msg, ...) void(0)
/**
 * Log a warning message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the message (ignored)
 */
#define LUCID_WARN(msg) void(0)
/**
 * Log a formatted warning message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the format string (ignored)
 * @param ... format arguments (ignored)
 */
#define LUCID_WARN_FMT(msg, ...) void(0)
/**
 * Log an error message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the message (ignored)
 */
#define LUCID_ERROR(msg) void(0)
/**
 * Log a formatted error message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the format string (ignored)
 * @param ... format arguments (ignored)
 */
#define LUCID_ERROR_FMT(msg, ...) void(0)
/**
 * Log a critical message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the message (ignored)
 */
#define LUCID_CRITICAL(msg) void(0)
/**
 * Log a formatted critical message (NLOG version).
 * When NLOG is defined, this is a no-op.
 * @param msg the format string (ignored)
 * @param ... format arguments (ignored)
 */
#define LUCID_CRITICAL_FMT(msg, ...) void(0)
/**
 * Check if trace logging is enabled (NLOG version).
 * When NLOG is defined, this always returns false.
 * @return always false
 */
#define LUCID_TRACE_ENABLED false
/**
 * Check if debug logging is enabled (NLOG version).
 * When NLOG is defined, this always returns false.
 * @return always false
 */
#define LUCID_DEBUG_ENABLED false
/**
 * Check if info logging is enabled (NLOG version).
 * When NLOG is defined, this always returns false.
 * @return always false
 */
#define LUCID_INFO_ENABLED false
/**
 * Check if warning logging is enabled (NLOG version).
 * When NLOG is defined, this always returns false.
 * @return always false
 */
#define LUCID_WARN_ENABLED false
/**
 * Check if error logging is enabled (NLOG version).
 * When NLOG is defined, this always returns false.
 * @return always false
 */
#define LUCID_ERROR_ENABLED false
/**
 * Check if critical logging is enabled (NLOG version).
 * When NLOG is defined, this always returns false.
 * @return always false
 */
#define LUCID_CRITICAL_ENABLED false

#endif
