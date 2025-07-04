/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/util/logging.h"

#ifndef NLOG

#include <spdlog/common.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/callback_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>

namespace lucid::log {

spdlog::level::level_enum level_ = spdlog::level::off;  ///< Default logging level is off. It can be set by the user.

std::shared_ptr<spdlog::logger> get_logger(const LoggerType logger_type) {
  // Checks if there exists a logger with the name. If it exists, return it.
  const char *logger_name = logger_type == LoggerType::OUT ? "lucid_out" : "lucid_err";
  // NOLINTNEXTLINE(build/include_what_you_use): false positive
  std::shared_ptr<spdlog::logger> logger{spdlog::get(logger_name)};
  if (logger) return logger;

  // Create and return a new logger.
  logger = logger_type == LoggerType::OUT ? spdlog::stdout_color_mt(logger_name) : spdlog::stderr_color_mt(logger_name);

  // Set the level to the one specified by the user or default to off.
  logger->set_level(level_);
  // Ensure that the logger flushes on error or critical messages.
  logger->flush_on(spdlog::level::err);

  // Set format.
  logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] %v");

  return logger;
}

void set_logger_sink(spdlog::custom_log_callback cb) {
  clear_logger();

  // Create a pair of loggers that will use the same callback sink.
  auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(cb);
  callback_sink->set_level(spdlog::level::trace);
  spdlog::register_logger(std::make_shared<spdlog::logger>("lucid_out", callback_sink));
  spdlog::register_logger(std::make_shared<spdlog::logger>("lucid_err", callback_sink));
}
void set_logger_sink(std::function<void(std::string)> cb) {
  // Format the log message so that we can use the resulting string in the callback.
  set_logger_sink([cb = std::move(cb)](const spdlog::details::log_msg &msg) {
    const auto formatter{spdlog::details::make_unique<spdlog::pattern_formatter>()};
    spdlog::memory_buf_t formatted;
    formatter->format(msg, formatted);
    cb(std::move(fmt::to_string(formatted)));
  });
}
void clear_logger() {
  // Drop existing loggers
  spdlog::drop_all();
}
void set_verbosity_level(const spdlog::level::level_enum level) {
  level_ = level;
  get_logger(LoggerType::OUT)->set_level(level);
  get_logger(LoggerType::ERR)->set_level(level);
}
}  // namespace lucid::log

#else

#endif  // NLOG
