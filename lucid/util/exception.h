/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Exception classes for lucid.
 */
#pragma once

#include <stdexcept>
#include <string>

/**
 * @namespace lucid::exception
 * Collection of exceptions that can be thrown by lucid.
 */
namespace lucid::exception {

/** Base class for all exceptions in lucid. */
class LucidException : public std::runtime_error {
 public:
  explicit LucidException(const char* const message) : std::runtime_error{message} {}
  explicit LucidException(const std::string& message) : std::runtime_error{message} {}
};

/** Exception for not yet implemented features. */
class LucidNotImplementedException final : public LucidException {
 public:
  LucidNotImplementedException() : LucidException("Not yet implemented") {}
  explicit LucidNotImplementedException(const char* const message) : LucidException{message} {}
  explicit LucidNotImplementedException(const std::string& message) : LucidException{message} {}
};

/** Exception for not supported features. */
class LucidNotSupportedException final : public LucidException {
 public:
  LucidNotSupportedException() : LucidException("Not supported") {}
  explicit LucidNotSupportedException(const char* const message) : LucidException{message} {}
  explicit LucidNotSupportedException(const std::string& message) : LucidException{message} {}
};

/** Exception for invalid arguments. */
class LucidInvalidArgumentException final : public LucidException, private std::invalid_argument {
 public:
  LucidInvalidArgumentException() : LucidException("Invalid argument"), std::invalid_argument{""} {}
  explicit LucidInvalidArgumentException(const char* const message)
      : LucidException{message}, std::invalid_argument{message} {}
  explicit LucidInvalidArgumentException(const std::string& message)
      : LucidException{message}, std::invalid_argument{message} {}
};

/** Exception for assertion failures. */
class LucidAssertionException final : public LucidException {
 public:
  explicit LucidAssertionException(const char* const message) : LucidException{message} {}
  explicit LucidAssertionException(const std::string& message) : LucidException{message} {}
};

/** Exception in the LP solver */
class LucidLpSolverException final : public LucidException {
 public:
  explicit LucidLpSolverException(const char* const message) : LucidException{message} {}
  explicit LucidLpSolverException(const std::string& message) : LucidException{message} {}
};

/** Exception for parser errors. */
class LucidParserException final : public LucidException {
 public:
  explicit LucidParserException(const char* const message) : LucidException{message} {}
  explicit LucidParserException(const std::string& message) : LucidException{message} {}
};

/** Exception for out of range errors. */
class LucidOutOfRangeException final : public LucidException, private std::out_of_range {
 public:
  explicit LucidOutOfRangeException(const char* const message) : LucidException{message}, std::out_of_range{message} {}
  explicit LucidOutOfRangeException(const std::string& message) : LucidException{message}, std::out_of_range{message} {}
};

/** Exception for unreachable code. */
class LucidUnreachableException final : public LucidException {
 public:
  LucidUnreachableException() : LucidException("Unreachable code") {}
  explicit LucidUnreachableException(const char* const message) : LucidException{message} {}
  explicit LucidUnreachableException(const std::string& message) : LucidException{message} {}
};

/** Exception occurred in Python code. */
class LucidPyException final : public LucidException {
 public:
  LucidPyException() : LucidException("Python exception") {}
  explicit LucidPyException(const char* const message) : LucidException{message} {}
  explicit LucidPyException(const std::string& message) : LucidException{message} {}
};

}  // namespace lucid::exception
