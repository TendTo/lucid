/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Utilities that verify assumptions made by the program and aborts
 * the program if those assumptions are not true.
 * If NDEBUG is defined, most of the macro do nothing and give no explanation.
 * It makes the program faster, but less useful for debugging.
 */
#pragma once

#include <fmt/core.h>

#include <stdexcept>

#include "lucid/util/exception.h"

#ifdef NDEBUG

#define LUCID_ASSERT(condition, msg) ((void)0)
#define LUCID_ASSERT_FMT(condition, msg, ...) ((void)0)
#define LUCID_UNREACHABLE() std::terminate()
#define LUCID_NOT_IMPLEMENTED() throw ::lucid::LucidNotImplementedException()
#define LUCID_RUNTIME_ERROR(msg) throw ::lucid::LucidException(msg)
#define LUCID_RUNTIME_ERROR_FMT(msg, ...) throw ::lucid::LucidException(fmt::format(msg, __VA_ARGS__))
#define LUCID_OUT_OF_RANGE(msg) throw ::lucid::LucidOutOfRangeException(msg)
#define LUCID_OUT_OF_RANGE_FMT(msg, ...) throw ::lucid::LucidOutOfRangeException(fmt::format(msg, __VA_ARGS__))
#define LUCID_INVALID_ARGUMENT(argument, actual) \
  throw ::lucid::LucidInvalidArgumentException(fmt::format("Invalid argument for {}: {}", argument, actual))
#define LUCID_INVALID_ARGUMENT_EXPECTED(argument, actual, expected) \
  throw ::lucid::LucidInvalidArgumentException(                      \
      fmt::format("Invalid argument for {}: received '{}', expected '{}'", argument, actual, expected))
#define LUCID_PY_ERROR(message) throw ::lucid::LucidPyException(message)
#define LUCID_PY_ERROR_FMT(msg, ...) throw ::lucid::LucidPyException(fmt::format(msg, __VA_ARGS__))

#else

#include "lucid/util/logging.h"

#define LUCID_ASSERT(condition, message)                                                                 \
  do {                                                                                                  \
    if (!(condition)) {                                                                                 \
      LUCID_CRITICAL_FMT("Assertion `{}` failed in {}:{}: {}", #condition, __FILE__, __LINE__, message); \
      throw ::lucid::LucidAssertionException(                                                             \
          fmt::format("Assertion `{}` failed in {}:{}: {}", #condition, __FILE__, __LINE__, message));  \
    }                                                                                                   \
  } while (false)

#define LUCID_ASSERT_FMT(condition, message, ...)                                                                  \
  do {                                                                                                            \
    if (!(condition)) {                                                                                           \
      LUCID_CRITICAL_FMT("Assertion `{}` failed in {}:{}\n" message, #condition, __FILE__, __LINE__, __VA_ARGS__); \
      throw ::lucid::LucidAssertionException(                                                                       \
          fmt::format("Assertion `{}` failed in {}:{}: " message, #condition, __FILE__, __LINE__, __VA_ARGS__));  \
    }                                                                                                             \
  } while (false)

#define LUCID_UNREACHABLE()                                                                                     \
  do {                                                                                                         \
    LUCID_CRITICAL_FMT("{}:{} Should not be reachable.", __FILE__, __LINE__);                                   \
    throw ::lucid::LucidUnreachableException(fmt::format("{}:{} Should not be reachable.", __FILE__, __LINE__)); \
  } while (false)

#define LUCID_NOT_IMPLEMENTED()                                                                            \
  do {                                                                                                    \
    LUCID_CRITICAL_FMT("{}:{} Not implemented.", __FILE__, __LINE__);                                      \
    throw ::lucid::LucidNotImplementedException(fmt::format("{}:{} Not implemented.", __FILE__, __LINE__)); \
  } while (false)

#define LUCID_RUNTIME_ERROR(msg)       \
  do {                                \
    LUCID_CRITICAL(msg);               \
    throw ::lucid::LucidException(msg); \
  } while (false)

#define LUCID_RUNTIME_ERROR_FMT(msg, ...)                        \
  do {                                                          \
    LUCID_CRITICAL_FMT(msg, __VA_ARGS__);                        \
    throw ::lucid::LucidException(fmt::format(msg, __VA_ARGS__)); \
  } while (false)

#define LUCID_OUT_OF_RANGE(msg)                  \
  do {                                          \
    LUCID_CRITICAL(msg);                         \
    throw ::lucid::LucidOutOfRangeException(msg); \
  } while (false)

#define LUCID_OUT_OF_RANGE_FMT(msg, ...)                                   \
  do {                                                                    \
    LUCID_CRITICAL_FMT(msg, __VA_ARGS__);                                  \
    throw ::lucid::LucidOutOfRangeException(fmt::format(msg, __VA_ARGS__)); \
  } while (false)

#define LUCID_INVALID_ARGUMENT(argument, actual) \
  throw ::lucid::LucidInvalidArgumentException(fmt::format("Invalid argument for {}: {}", argument, actual))

#define LUCID_INVALID_ARGUMENT_EXPECTED(argument, actual, expected) \
  throw ::lucid::LucidInvalidArgumentException(                      \
      fmt::format("Invalid argument for {}: received '{}', expected '{}'", argument, actual, expected))

#define LUCID_PY_ERROR(msg)              \
  do {                                  \
    LUCID_CRITICAL(msg);                 \
    PyErr_Print();                      \
    throw ::lucid::LucidPyException(msg); \
  } while (false)

#define LUCID_PY_ERROR_FMT(msg, ...)                               \
  do {                                                            \
    LUCID_CRITICAL_FMT(msg, __VA_ARGS__);                          \
    PyErr_Print();                                                \
    throw ::lucid::LucidPyException(fmt::format(msg, __VA_ARGS__)); \
  } while (false)

#define LUCID_NOT_SUPPORTED(msg)                                                        \
  do {                                                                                 \
    LUCID_CRITICAL_FMT("{} is not supported.", msg);                                    \
    throw ::lucid::LucidNotSupportedException(fmt::format("{} is not supported.", msg)); \
  } while (false)

#endif  // NDEBUG
