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

/**
 * Log an error message and throw an exception.
 * This macro logs the formatted message at ERROR level and then throws the specified exception.
 * @param ex the exception type to throw (without lucid::exception:: prefix)
 * @param msg the format string for the error message
 * @param ... arguments to format into the message
 * @throw ex the specified exception with the formatted message
 */
#define LUCID_ERROR_LOG_AND_THROW(ex, msg, ...)                  \
  do {                                                           \
    LUCID_ERROR_FMT(msg, __VA_ARGS__);                           \
    throw ::lucid::exception::ex(fmt::format(msg, __VA_ARGS__)); \
  } while (false)

/**
 * Log a critical message and throw an exception.
 * This macro logs the formatted message at CRITICAL level and then throws the specified exception.
 * @param ex the exception type to throw (without lucid::exception:: prefix)
 * @param msg the format string for the critical message
 * @param ... arguments to format into the message
 * @throw ex the specified exception with the formatted message
 */
#define LUCID_CRITICAL_LOG_AND_THROW(ex, msg, ...)               \
  do {                                                           \
    LUCID_CRITICAL_FMT(msg, __VA_ARGS__);                        \
    throw ::lucid::exception::ex(fmt::format(msg, __VA_ARGS__)); \
  } while (false)

#ifndef NDEBUG

/**
 * Assert that a condition is true, throwing an exception if it fails.
 * This macro checks the condition and throws a LucidAssertionException with file and line information if the condition
 * is false. Only active when NDEBUG is not defined (debug builds).
 * @param condition the condition to check
 * @param message descriptive message explaining what the assertion checks
 * @throw LucidAssertionException if the condition is false
 */
#define LUCID_ASSERT(condition, message)                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      LUCID_CRITICAL_LOG_AND_THROW(LucidAssertionException, "Assertion `{}` failed in {}:{}: {}", condition, __FILE__, \
                                   __LINE__, message);                                                                 \
    }                                                                                                                  \
  } while (false)

/**
 * Mark code that should never be reached, throwing an exception if executed.
 * This macro throws a LucidUnreachableException with file and line information.
 * Only active when NDEBUG is not defined (debug builds).
 * @throw LucidUnreachableException always
 */
#define LUCID_UNREACHABLE() \
  LUCID_CRITICAL_LOG_AND_THROW(LucidUnreachableException, "{}:{} Should not be reachable.", __FILE__, __LINE__)

#else

#define LUCID_ASSERT(condition, msg) ((void)0)
#define LUCID_UNREACHABLE() std::terminate()

#endif

#ifndef NCHECK

/**
 * Check that an argument satisfies a condition, throwing an exception if it doesn't.
 * This macro validates function arguments and throws a LucidInvalidArgumentException if the condition is false.
 * Only active when NCHECK is not defined.
 * @param condition the condition that must be true
 * @param argument name of the argument being checked
 * @param actual the actual value of the argument
 * @throw LucidInvalidArgumentException if the condition is false
 */
#define LUCID_CHECK_ARGUMENT(condition, argument, actual)                                                          \
  do {                                                                                                             \
    if (!(condition)) {                                                                                            \
      LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException, "Invalid argument for {}: '{}'", argument, actual); \
    }                                                                                                              \
  } while (false)
/**
 * Check that an argument satisfies a condition with expected value information.
 * This macro validates function arguments and provides both actual and expected values in the error message.
 * Only active when NCHECK is not defined.
 * @param condition the condition that must be true
 * @param argument name of the argument being checked
 * @param actual the actual value of the argument
 * @param expected the expected value or description
 * @throw LucidInvalidArgumentException if the condition is false
 */
#define LUCID_CHECK_ARGUMENT_EXPECTED(condition, argument, actual, expected)                                          \
  do {                                                                                                                \
    if (!(condition)) {                                                                                               \
      LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException,                                                        \
                                "Invalid argument for {}: received '{}', expected '{}'", argument, actual, expected); \
    }                                                                                                                 \
  } while (false)
/**
 * Check that a value satisfies a comparison operation with an expected value.
 * This macro validates that value op expected is true, where op is a comparison operator.
 * Only active when NCHECK is not defined.
 * @param value the value to check
 * @param op the comparison operator (e.g., ==, !=, <, >, <=, >=)
 * @param expected the expected value for comparison
 * @throw LucidInvalidArgumentException if the comparison fails
 */
#define LUCID_CHECK_ARGUMENT_CMP(value, op, expected)                                                           \
  do {                                                                                                          \
    if (!((value)op(expected))) {                                                                               \
      LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException,                                                  \
                                "Invalid argument " #value " violates constraint " #value " " #op " " #expected \
                                " : received '{}', expected '" #op " {}'",                                      \
                                value, expected);                                                               \
    }                                                                                                           \
  } while (false)
/**
 * Check that a value equals an expected value.
 * This macro validates that value == expected is true.
 * Only active when NCHECK is not defined.
 * @param value the value to check
 * @param expected the expected value
 * @throw LucidInvalidArgumentException if value != expected
 */
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

/**
 * Throw a LucidNotImplementedException when functionality is not yet implemented.
 * This macro logs an error and throws an exception with file and line information.
 * @throw LucidNotImplementedException always
 */
#define LUCID_NOT_IMPLEMENTED() \
  LUCID_ERROR_LOG_AND_THROW(LucidNotImplementedException, "{}:{} Not implemented.", __FILE__, __LINE__)

/**
 * Throw a LucidException with a simple message.
 * This macro logs an error and throws a general LucidException.
 * @param msg the error message
 * @throw LucidException always
 */
#define LUCID_RUNTIME_ERROR(msg) LUCID_ERROR_LOG_AND_THROW(LucidException, msg, "")

/**
 * Throw a LucidException with a formatted message.
 * This macro logs an error and throws a general LucidException with formatted arguments.
 * @param msg the format string for the error message
 * @param ... arguments to format into the message
 * @throw LucidException always
 */
#define LUCID_RUNTIME_ERROR_FMT(msg, ...) LUCID_ERROR_LOG_AND_THROW(LucidException, msg, __VA_ARGS__)

/**
 * Throw a LucidOutOfRangeException with a simple message.
 * This macro logs an error and throws an out-of-range exception.
 * @param msg the error message
 * @throw LucidOutOfRangeException always
 */
#define LUCID_OUT_OF_RANGE(msg) LUCID_ERROR_LOG_AND_THROW(LucidOutOfRangeException, msg, "")

/**
 * Throw a LucidOutOfRangeException with a formatted message.
 * This macro logs an error and throws an out-of-range exception with formatted arguments.
 * @param msg the format string for the error message
 * @param ... arguments to format into the message
 * @throw LucidOutOfRangeException always
 */
#define LUCID_OUT_OF_RANGE_FMT(msg, ...) LUCID_ERROR_LOG_AND_THROW(LucidOutOfRangeException, msg, __VA_ARGS__)

/**
 * Log an error message showing the current value and throw a LucidInvalidArgumentException.
 * @param argument hyperparameter that is invalid
 * @param actual actual value of the hyperparameter
 */
#define LUCID_INVALID_ARGUMENT(argument, actual) \
  LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException, "Invalid argument for {}: {}", argument, actual)

/**
 * Log an error message, showing the expected value and throw a LucidInvalidArgumentException.
 * @param argument hyperparameter that is invalid
 * @param actual actual value of the hyperparameter
 * @param expected expected value of the hyperparameter
 */
#define LUCID_INVALID_ARGUMENT_EXPECTED(argument, actual, expected)                                                 \
  LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException, "Invalid argument for {}: received '{}', expected '{}'", \
                            argument, actual, expected)

/**
 * Log an error message and throw a LucidInvalidArgumentException.
 * Used when a hyperparameter is invalid has an invalid type.
 * @param parameter hyperparameter that is invalid
 * @param type type of the hyperparameter
 */
#define LUCID_INVALID_HYPER_PARAMETER(parameter, type) \
  LUCID_ERROR_LOG_AND_THROW(LucidInvalidArgumentException, "Invalid hyper parameter {} of type {}", parameter, type)

/**
 * Log an error message and throw a LucidPyException.
 * @param msg message to log
 */
#define LUCID_PY_ERROR(msg)                          \
  do {                                               \
    LUCID_ERROR(msg);                                \
    PyErr_Print();                                   \
    throw ::lucid::exception::LucidPyException(msg); \
  } while (false)

/**
 * Log an error message and throw a LucidPyException.
 * @param msg  error message to log
 */
#define LUCID_PY_ERROR_FMT(msg, ...)                                           \
  do {                                                                         \
    LUCID_ERROR_FMT(msg, __VA_ARGS__);                                         \
    PyErr_Print();                                                             \
    throw ::lucid::exception::LucidPyException(fmt::format(msg, __VA_ARGS__)); \
  } while (false)

/**
 * Throw a LucidNotSupportedException when a functionality is not supported.
 * @param msg functionality or feature that is not supported
 */
#define LUCID_NOT_SUPPORTED(msg) LUCID_ERROR_LOG_AND_THROW(LucidNotSupportedException, "{} is not supported.", msg)

/**
 * Throw a LucidNotSupportedException when a functionality is not supported because the software was compiled without
 * a required dependency.
 * @param msg functionality or feature that is not supported
 * @param dependency missing runtime dependency that is required for the functionality to work
 * @throw LucidNotSupportedException
 */
#define LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY(msg, dependency)                                         \
  LUCID_ERROR_LOG_AND_THROW(                                                                                  \
      LucidNotSupportedException,                                                                             \
      "{} is not supported because the following dependency was not included during compilation: '{}'.", msg, \
      dependency)

/**
 * Throw a LucidNotSupportedException when a functionality is not supported because of a missing runtime dependency
 * @param msg functionality or feature that is not supported.
 * @param dependency missing runtime dependency that is required for the functionality to work
 * @throw LucidNotSupportedException
 */
#define LUCID_NOT_SUPPORTED_MISSING_RUNTIME_DEPENDENCY(msg, dependency) \
  LUCID_ERROR_LOG_AND_THROW(                                            \
      LucidNotSupportedException,                                       \
      "{} is not supported because the following dependency was not found on this system: '{}'.", msg, dependency)
