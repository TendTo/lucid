/**
 * @author lucid_authors
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

#define LUCID_ERROR_LOG_AND_THROW(ex, msg, ...)                  \
  do {                                                           \
    LUCID_ERROR_FMT(msg, __VA_ARGS__);                           \
    throw ::lucid::exception::ex(fmt::format(msg, __VA_ARGS__)); \
  } while (false)

#define LUCID_CRITICAL_LOG_AND_THROW(ex, msg, ...)               \
  do {                                                           \
    LUCID_CRITICAL_FMT(msg, __VA_ARGS__);                        \
    throw ::lucid::exception::ex(fmt::format(msg, __VA_ARGS__)); \
  } while (false)

#ifndef NDEBUG

#define LUCID_ASSERT(condition, message)                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      LUCID_CRITICAL_LOG_AND_THROW(LucidAssertionException, "Assertion `{}` failed in {}:{}: {}", condition, __FILE__, \
                                   __LINE__, message);                                                                 \
    }                                                                                                                  \
  } while (false)

#define LUCID_UNREACHABLE() \
  LUCID_CRITICAL_LOG_AND_THROW(LucidUnreachableException, "{}:{} Should not be reachable.", __FILE__, __LINE__)

#else

#define LUCID_ASSERT(condition, msg) ((void)0)
#define LUCID_UNREACHABLE() std::terminate()

#endif

#ifndef NCHECK

#define LUCID_CHECK_ARGUMENT(condition, argument, actual)                                                          \
  do {                                                                                                             \
    if (!(condition)) {                                                                                            \
      LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException, "Invalid argument for {}: '{}'", argument, actual); \
    }                                                                                                              \
  } while (false)
#define LUCID_CHECK_ARGUMENT_EXPECTED(condition, argument, actual, expected)                                          \
  do {                                                                                                                \
    if (!(condition)) {                                                                                               \
      LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException,                                                        \
                                "Invalid argument for {}: received '{}', expected '{}'", argument, actual, expected); \
    }                                                                                                                 \
  } while (false)
#define LUCID_CHECK_ARGUMENT_CMP(value, op, expected)                                                      \
  do {                                                                                                     \
    if (!((value)op(expected))) {                                                                          \
      LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException,                                             \
                                "Invalid argument " #value " violates constraint " #value " == " #expected \
                                " : received '{}', expected '" #op " {}'",                                 \
                                value, expected);                                                          \
    }                                                                                                      \
  } while (false)
#define LUCID_CHECK_ARGUMENT_EQ(value, expected)                                                           \
  do {                                                                                                     \
    if (!((value) == (expected))) {                                                                        \
      LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException,                                             \
                                "Invalid argument " #value " violates constraint " #value " == " #expected \
                                " : received '{}', expected '{}'",                                         \
                                value, expected);                                                          \
    }                                                                                                      \
  } while (false)

#else

#define LUCID_CHECK_ARGUMENT(condition, argument, actual) ((void)0)
#define LUCID_CHECK_ARGUMENT_EXPECTED(condition, argument, actual, expected) ((void)0)
#define LUCID_CHECK_ARGUMENT_CMP(value, op, expected) ((void)0)
#define LUCID_CHECK_ARGUMENT_EQ(value, expected) ((void)0)

#endif  // NCHECK

#define LUCID_NOT_IMPLEMENTED() \
  LUCID_ERROR_LOG_AND_THROW(LucidNotImplementedException, "{}:{} Not implemented.", __FILE__, __LINE__)

#define LUCID_RUNTIME_ERROR(msg) LUCID_ERROR_LOG_AND_THROW(LucidException, msg, "")

#define LUCID_RUNTIME_ERROR_FMT(msg, ...) LUCID_ERROR_LOG_AND_THROW(LucidException, msg, __VA_ARGS__)

#define LUCID_OUT_OF_RANGE(msg) LUCID_ERROR_LOG_AND_THROW(LucidOutOfRangeException, msg, "")

#define LUCID_OUT_OF_RANGE_FMT(msg, ...) LUCID_ERROR_LOG_AND_THROW(LucidOutOfRangeException, msg, __VA_ARGS__)

#define LUCID_INVALID_ARGUMENT(argument, actual) \
  LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException, "Invalid argument for {}: {}", argument, actual)

#define LUCID_INVALID_ARGUMENT_EXPECTED(argument, actual, expected)                                                 \
  LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException, "Invalid argument for {}: received '{}', expected '{}'", \
                            argument, actual, expected)

#define LUCID_INVALID_HYPER_PARAMETER(parameter, type) \
  LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException, "Invalid hyper parameter {} of type {}", parameter, type)

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

#define LUCID_NOT_SUPPORTED(msg) LUCID_ERROR_LOG_AND_THROW(LucidNotSupportedException, "{} is not supported.", msg)

#define LUCID_NOT_SUPPORTED_MISSING_DEPENDENCY(msg, dependency)                                               \
  LUCID_ERROR_LOG_AND_THROW(                                                                                  \
      LucidNotSupportedException,                                                                             \
      "{} is not supported because the following dependency was not included during compilation: '{}'.", msg, \
      dependency)
