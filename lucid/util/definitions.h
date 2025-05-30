/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Utility definitions.
 */
#pragma once

#define FLAG_ENUMS(name, max_enum)                                                                     \
  /**                                                                                                  \
   * Perform a bitwise OR operation on two name##.                                                     \
   * Efficient way of taking the union of two set of name##.                                           \
   * @tparam LP left name type, must be one of the `name` enum values or a set of name## \             \
   * @tparam RP right name type, must be one of the `name` enum values or a set of name##              \
   * @param lhs left-hand side name                                                                    \
   * @param rhs right-hand side name                                                                   \
   * @return the result of the bitwise OR operation as a `name##` \                                    \
   */                                                                                                  \
  template <IsAnyOf<name, name##s> LP, IsAnyOf<name, name> RP>                                         \
  constexpr name operator|(LP lhs, RP rhs) {                                                           \
    return static_cast<name>(lhs) | static_cast<name>(rhs);                                            \
  }                                                                                                    \
  /**                                                                                                  \
   * Perform a bitwise OR operation on two name## and return the result as a boolean.                  \
   * Efficient way of checking if two set of name## have a non-empty union.                            \
   * @code                                                                                             \
   * name::SIGMA_L || name::SIGMA_L; // true                                                           \
   * name::SIGMA_F || name::DEGREE; // true                                                            \
   * name::SIGMA_F || (name::DEGREE & name::SIGMA_F); // true                                          \
   * name::_ || name::SIGMA_L; // true                                                                 \
   * name::_ || name::_; // false                                                                      \
   * @endcode                                                                                          \
   * @tparam LP left name type, must be one of the `name` enum values or a set of name##               \
   * @tparam RP right name type, must be one of the `name` enum values or a set of name##              \
   * @param lhs left-hand side name                                                                    \
   * @param rhs right-hand side name                                                                   \
   * @return the result of the bitwise OR operation as a boolean                                       \
   */                                                                                                  \
  template <IsAnyOf<name, name> LP, IsAnyOf<name, name> RP>                                            \
  constexpr bool operator||(LP lhs, RP rhs) {                                                          \
    return static_cast<name>(lhs) | static_cast<name>(rhs);                                            \
  }                                                                                                    \
  /**                                                                                                  \
   * Perform a bitwise AND operation on two name##.                                                    \
   * Efficient way of taking the intersection of two set of name##.                                    \
   * @tparam LP left name type, must be one of the `name` enum values or a set of name##               \
   * @tparam RP right name type, must be one of the `name` enum values or a set of name##              \
   * @param lhs left-hand side name                                                                    \
   * @param rhs right-hand side name                                                                   \
   * @return the result of the bitwise AND operation as a `name`                                       \
   */                                                                                                  \
  template <IsAnyOf<name, name> LP, IsAnyOf<name, name> RP>                                            \
  constexpr name operator&(LP lhs, RP rhs) {                                                           \
    return static_cast<name>(lhs) & static_cast<name>(rhs);                                            \
  }                                                                                                    \
  /**                                                                                                  \
   * Perform a bitwise AND operation on two name## and return the result as a boolean.                 \
   * Efficient way of checking if two set of name## have a non-empty intersection.                     \
   * @code                                                                                             \
   * name::SIGMA_L && name::SIGMA_L; // true                                                           \
   * name::SIGMA_F && name::DEGREE; // false                                                           \
   * name::SIGMA_F && (name::DEGREE & name::SIGMA_F); // true                                          \
   * name::_ && name::SIGMA_L; // false                                                                \
   * name::_ && name::_; // false                                                                      \
   * @endcode                                                                                          \
   * @tparam LP left name type, must be one of the `name` enum values or a set of name##               \
   * @tparam RP right name type, must be one of the `name` enum values or a set of name##              \
   * @param lhs left-hand side name                                                                    \
   * @param rhs right-hand side name                                                                   \
   * @return the result of the bitwise AND operation as a boolean                                      \
   */                                                                                                  \
  template <IsAnyOf<name, name> LP, IsAnyOf<name, name> RP>                                            \
  constexpr bool operator&&(LP lhs, RP rhs) {                                                          \
    return static_cast<name>(lhs) & static_cast<name>(rhs);                                            \
  }                                                                                                    \
  /**                                                                                                  \
   * Convert the efficient name representation into an easy-to-traverse vector                         \
   * @param name## set of name##                                                                       \
   * @return vector of name##                                                                          \
   */                                                                                                  \
  inline operator std::vector<name>(const name##s values) {                                            \
    std::vector<name> result;                                                                          \
    for (auto p = name::_; p <= name::max_enum; p = static_cast<name>(static_cast<name##s>(p) << 1)) { \
      if (p && values) result.push_back(p);                                                            \
    }                                                                                                  \
    return result;                                                                                     \
  }
