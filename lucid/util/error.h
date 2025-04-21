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
#include "lucid/util/logging.h"

#ifndef NDEBUG

#define LUCID_ASSERT(condition, message)                                                                 \
  do {                                                                                                   \
    if (!(condition)) {                                                                                  \
      LUCID_CRITICAL_FMT("Assertion `{}` failed in {}:{}: {}", #condition, __FILE__, __LINE__, message); \
      throw ::lucid::exception::LucidAssertionException(                                                 \
          fmt::format("Assertion `{}` failed in {}:{}: {}", #condition, __FILE__, __LINE__, message));   \
    }                                                                                                    \
  } while (false)

#define LUCID_ASSERT_FMT(condition, message, ...)                                                                  \
  do {                                                                                                             \
    if (!(condition)) {                                                                                            \
      LUCID_CRITICAL_FMT("Assertion `{}` failed in {}:{}\n" message, #condition, __FILE__, __LINE__, __VA_ARGS__); \
      throw ::lucid::exception::LucidAssertionException(                                                           \
          fmt::format("Assertion `{}` failed in {}:{}: " message, #condition, __FILE__, __LINE__, __VA_ARGS__));   \
    }                                                                                                              \
  } while (false)

#define LUCID_UNREACHABLE()                                                   \
  do {                                                                        \
    LUCID_CRITICAL_FMT("{}:{} Should not be reachable.", __FILE__, __LINE__); \
    throw ::lucid::exception::LucidUnreachableException(                      \
        fmt::format("{}:{} Should not be reachable.", __FILE__, __LINE__));   \
  } while (false)

#else

#define LUCID_ASSERT(condition, msg) ((void)0)
#define LUCID_ASSERT_FMT(condition, msg, ...) ((void)0)
#define LUCID_UNREACHABLE() std::terminate()

#endif

#ifndef NCHECK

#define LUCID_CHECK_ARGUMENT(condition, argument, actual)                  \
  do {                                                                     \
    if (!(condition)) {                                                    \
      LUCID_ERROR_FMT("Invalid argument for {}: '{}'", argument, actual);  \
      throw ::lucid::exception::LucidInvalidArgumentException(             \
          fmt::format("Invalid argument for {}: '{}'", argument, actual)); \
    }                                                                      \
  } while (false)
#define LUCID_CHECK_ARGUMENT_EXPECTED(condition, argument, actual, expected)                                 \
  do {                                                                                                       \
    if (!(condition)) {                                                                                      \
      LUCID_ERROR_FMT("Invalid argument for {}: received '{}', expected '{}'", argument, actual, expected);  \
      throw ::lucid::exception::LucidInvalidArgumentException(                                               \
          fmt::format("Invalid argument for {}: received '{}', expected '{}'", argument, actual, expected)); \
    }                                                                                                        \
  } while (false)

#else

#define LUCID_CHECK_ARGUMENT(condition, argument, actual) ((void)0)
#define LUCID_CHECK_ARGUMENT_EXPECTED(condition, argument, actual, expected) ((void)0)

#endif  // NCHECK

#define LUCID_NOT_IMPLEMENTED()                                                                                        \
  do {                                                                                                                 \
    LUCID_ERROR_FMT("{}:{} Not implemented.", __FILE__, __LINE__);                                                     \
    throw ::lucid::exception::LucidNotImplementedException(fmt::format("{}:{} Not implemented.", __FILE__, __LINE__)); \
  } while (false)

#define LUCID_RUNTIME_ERROR(msg)                   \
  do {                                             \
    LUCID_ERROR(msg);                              \
    throw ::lucid::exception::LucidException(msg); \
  } while (false)

#define LUCID_RUNTIME_ERROR_FMT(msg, ...)                                    \
  do {                                                                       \
    LUCID_ERROR_FMT(msg, __VA_ARGS__);                                       \
    throw ::lucid::exception::LucidException(fmt::format(msg, __VA_ARGS__)); \
  } while (false)

#define LUCID_OUT_OF_RANGE(msg)                              \
  do {                                                       \
    LUCID_ERROR(msg);                                        \
    throw ::lucid::exception::LucidOutOfRangeException(msg); \
  } while (false)

#define LUCID_OUT_OF_RANGE_FMT(msg, ...)                                               \
  do {                                                                                 \
    LUCID_ERROR_FMT(msg, __VA_ARGS__);                                                 \
    throw ::lucid::exception::LucidOutOfRangeException(fmt::format(msg, __VA_ARGS__)); \
  } while (false)

#define LUCID_INVALID_ARGUMENT(argument, actual)                       \
  do {                                                                 \
    LUCID_ERROR_FMT("Invalid argument for {}: {}", argument, actual);  \
    throw ::lucid::exception::LucidInvalidArgumentException(           \
        fmt::format("Invalid argument for {}: {}", argument, actual)); \
  } while (false)

#define LUCID_INVALID_ARGUMENT_EXPECTED(argument, actual, expected)                                        \
  do {                                                                                                     \
    LUCID_ERROR_FMT("Invalid argument for {}: received '{}', expected '{}'", argument, actual, expected);  \
    throw ::lucid::exception::LucidInvalidArgumentException(                                               \
        fmt::format("Invalid argument for {}: received '{}', expected '{}'", argument, actual, expected)); \
  } while (false)

#define LUCID_PY_ERROR(msg)                          \
  do {                                               \
    LUCID_ERROR(msg);                                \
    PyErr_Print();                                   \
    throw ::lucid::exception::LucidPyException(msg); \
  } while (false)

#define LUCID_PY_ERROR_FMT(msg, ...)                                           \
  do {                                                                         \
    LUCID_ERROR_FMT(msg, __VA_ARGS__);                                         \
    PyErr_Print();                                                             \
    throw ::lucid::exception::LucidPyException(fmt::format(msg, __VA_ARGS__)); \
  } while (false)

#define LUCID_NOT_SUPPORTED(msg)                                                                    \
  do {                                                                                              \
    LUCID_ERROR_FMT("{} is not supported.", msg);                                                   \
    throw ::lucid::exception::LucidNotSupportedException(fmt::format("{} is not supported.", msg)); \
  } while (false)

#define LUCID_NOT_SUPPORTED_MISSING_DEPENDENCY(msg, dependency)                                                        \
  do {                                                                                                                 \
    LUCID_ERROR_FMT("{} is not supported because the following dependency was not included during compilation: '{}'.", \
                    msg, dependency);                                                                                  \
    throw ::lucid::exception::LucidNotSupportedException(                                                              \
        fmt::format("{} is not supported because the following dependency was not included during compilation: '{}'.", \
                    msg, dependency));                                                                                 \
  } while (false)
