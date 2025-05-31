/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Fast bitwise operations on enums.
 */
#pragma once

/**
 * Fast bitwise operations on enums.
 * @todo For clarity, once the design is finalised, consider moving all the definitions in the correct enum class files.
 * @param name enum name, e.g., `Parameter`
 * @param names underlying type of the enum, e.g., `Parameters`
 * @param max_enum maximum enum value, e.g., `DEGREE`
 */
#define LUCID_FLAG_ENUMS(name, names, max_enum)                                                      \
  /**                                                                                                \
   * Perform a bitwise OR operation on two name##.                                                   \
   * Efficient way of taking the union of two set of name##.                                         \
   * @tparam L left name type, must be one of the `name` enum values or a set of name## \            \
   * @tparam R right name type, must be one of the `name` enum values or a set of name##             \
   * @param lhs left-hand side name                                                                  \
   * @param rhs right-hand side name                                                                 \
   * @return the result of the bitwise OR operation as a `name##`                                    \
   */                                                                                                \
  template <IsAnyOf<name, names> L, IsAnyOf<name, names> R>                                          \
  constexpr names operator|(L lhs, R rhs)                                                            \
    requires(!(std::is_same_v<L, R> && std::is_same_v<L, names>))                                    \
  {                                                                                                  \
    return static_cast<names>(lhs) | static_cast<names>(rhs);                                        \
  }                                                                                                  \
  /**                                                                                                \
   * Perform a bitwise OR operation on two name## and return the result as a boolean.                \
   * Efficient way of checking if two set of name## have a non-empty union.                          \
   * @tparam L left name type, must be one of the `name` enum values or a set of name##              \
   * @tparam R right name type, must be one of the `name` enum values or a set of name##             \
   * @param lhs left-hand side name                                                                  \
   * @param rhs right-hand side name                                                                 \
   * @return the result of the bitwise OR operation as a boolean                                     \
   */                                                                                                \
  template <IsAnyOf<name, names> L, IsAnyOf<name, names> R>                                          \
  constexpr bool operator||(L lhs, R rhs) {                                                          \
    return static_cast<names>(lhs) | static_cast<names>(rhs);                                        \
  }                                                                                                  \
  /**                                                                                                \
   * Perform a bitwise AND operation on two name##.                                                  \
   * Efficient way of taking the intersection of two set of name##.                                  \
   * @tparam L left name type, must be one of the `name` enum values or a set of name##              \
   * @tparam R right name type, must be one of the `name` enum values or a set of name##             \
   * @param lhs left-hand side name                                                                  \
   * @param rhs right-hand side name                                                                 \
   * @return the result of the bitwise AND operation as a `name`                                     \
   */                                                                                                \
  template <IsAnyOf<name, names> L, IsAnyOf<name, names> R>                                          \
  constexpr names operator&(L lhs, R rhs)                                                            \
    requires(!(std::is_same_v<L, R> && std::is_same_v<L, names>))                                    \
  {                                                                                                  \
    return static_cast<names>(lhs) & static_cast<names>(rhs);                                        \
  }                                                                                                  \
  /**                                                                                                \
   * Perform a bitwise AND operation on two name## and return the result as a boolean.               \
   * Efficient way of checking if two set of name## have a non-empty intersection.                   \
   * @tparam L left name type, must be one of the `name` enum values or a set of name##              \
   * @tparam R right name type, must be one of the `name` enum values or a set of name##             \
   * @param lhs left-hand side name                                                                  \
   * @param rhs right-hand side name                                                                 \
   * @return the result of the bitwise AND operation as a boolean                                    \
   */                                                                                                \
  template <IsAnyOf<name, names> L, IsAnyOf<name, names> R>                                          \
  constexpr bool operator&&(L lhs, R rhs) {                                                          \
    return static_cast<names>(lhs) & static_cast<names>(rhs);                                        \
  }                                                                                                  \
  /**                                                                                                \
   * Convert the efficient name representation into an easy-to-traverse vector                       \
   * @param values set of name##                                                                     \
   * @return vector of name##                                                                        \
   */                                                                                                \
  inline std::vector<name> name##s_to_vector(const names values) {                                   \
    std::vector<name> result;                                                                        \
    for (auto p = name::_; p <= name::max_enum; p = static_cast<name>(static_cast<names>(p) << 1)) { \
      if (p && values) result.push_back(p);                                                          \
    }                                                                                                \
    return result;                                                                                   \
  }
