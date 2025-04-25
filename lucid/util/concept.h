/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Collection of concepts used in the lucid library.
 * Concepts have been introduced in the c++20 standard and are used in the lucid library
 * in order to make the code more readable and to provide better error messages in templated code.
 */
#pragma once

#include <concepts>

namespace lucid {

/**
 * Check if the type `T` is an object that has a `hash()` method.
 * The `hash()` method must return a value convertible to `std::size_t`.
 * @code
 * template <Hashable T>
 * void foo(T t); // T has to have a hash() method returning an std::size_t
 * @endcode
 * @tparam T type to check
 */
template <class T>
concept Hashable = requires(T t) {
  { t.hash() } -> std::convertible_to<std::size_t>;
};  // NOLINT(readability/braces) per C++ standard concept definition

/**
 * Check if the type `T` is a data container of type `U` having a `data()` method which returns a raw pointer
 * and a `size()` method which returns the size of the container.
 * @code
 * template <class U, SizedDataContainer<U> T>
 * void foo(T t); // T can be a vector or any other data container of type U with a data() and size() method
 * @endcode
 * @tparam T type to check
 * @tparam U type of the elements
 */
template <class T, class U>
concept SizedDataContainer = requires(T t) {
  { t.data() } -> std::convertible_to<U*>;
  { t.size() } -> std::convertible_to<std::size_t>;
};  // NOLINT(readability/braces) per C++ standard concept definition

/**
 * Check if the type `T` constitutes a map from type `From` to type `To`.
 * It has to have a method `at` that takes a `From` and returns a `To`.
 * @code
 * template <MapFromTo<int, std::string> T>
 * void foo(T t); // T can be a map from int to std::string or a vector of std::string
 * @endcode
 * @tparam T type to check
 * @tparam From type domain
 * @tparam To type codomain
 */
template <class T, class From, class To>
concept MapFromTo = requires(T t) {
  { t.at(From{}) } -> std::convertible_to<To>;
};  // NOLINT(readability/braces) per C++ standard concept definition

/**
 * Check if the type `T` is a self-reference counter type.
 * A self-reference counter type must have both the `AddRef` and `Release` methods.
 * The `AddRef` method should increment the internal reference counter.
 * The `Release` method should decrement it, and if it reaches zero, the object should be deleted using its destructor.
 * @code
 * template <SelfReferenceCounter T>
 * void foo(T t); // T can be any self-reference counter type
 * @endcode
 * @tparam T type to check
 */
template <class T>
concept SelfReferenceCounter = requires(T t) {
  { t.add_ref() } -> std::same_as<void>;
  { t.release() } -> std::same_as<void>;
};  // NOLINT(readability/braces) per C++ standard concept definition

/**
 * Check if the type `T` is an iterable type.
 * @code
 * template <Iterable T>
 * void foo(T t); // T can be any iterable type
 * @endcode
 * @tparam T type to check
 */
template <class T>
concept Iterable = requires(T t) {
  { t.begin() } -> std::convertible_to<typename T::iterator>;
  { t.end() } -> std::convertible_to<typename T::iterator>;
};  // NOLINT(readability/braces) per C++ standard concept definition

/**
 * Check if the type `T` is an iterable type with a `size()` method.
 * @code
 * template <SizedIterable T>
 * void foo(T t); // T can be any iterable type with a size() method
 * @endcode
 * @tparam T type to check
 */
template <class T>
concept SizedIterable = requires(T t) {
  { t.begin() } -> std::convertible_to<typename T::iterator>;
  { t.end() } -> std::convertible_to<typename T::iterator>;
  { t.size() } -> std::convertible_to<std::size_t>;
};  // NOLINT(readability/braces) per C++ standard concept definition

/**
 * Check if the type `T` is an iterable type with elements of type `U`.
 * @code
 * template <TypedIterable<int> T>
 * void foo(T t); // T can be any iterable type with elements of type int
 * @endcode
 * @tparam T type to check
 * @tparam U type of the elements
 */
template <class T, class U>
concept TypedIterable = requires(T t, U u) {
  { t.begin() } -> std::convertible_to<typename T::iterator>;
  { t.end() } -> std::convertible_to<typename T::iterator>;
} && std::convertible_to<typename T::value_type, U>;

/**
 * Check if the type `T` is an iterable type with elements of type `U` and a `size()` method.
 * @code
 * template <TypedSizedIterable<int> T>
 * void foo(T t); // T can be any iterable type with elements of type int and a size() method
 * @endcode
 * @tparam T type to check
 * @tparam U type of the elements
 */
template <class T, class U>
concept SizedTypedIterable = requires(T t, U u) {
  { t.begin() } -> std::convertible_to<typename T::iterator>;
  { t.end() } -> std::convertible_to<typename T::iterator>;
  { t.size() } -> std::convertible_to<std::size_t>;
} && std::convertible_to<typename T::value_type, U>;

/**
 * Check if the type `T` is any of the types `U`.
 * @code
 * template <IsAnyOf<int, float, bool> T>
 * void foo(T t); // T can be either int, float or bool
 * @endcode
 * @tparam T type to check
 * @tparam U any number of types to check against
 */
template <class T, class... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

/**
 * Check if the type `T` is not any of the types `U`.
 * @code
 * template <IsNotAnyOf<int, float, bool> T>
 * void foo(T t); // T can be any type except int, float or bool
 * @endcode
 * @tparam T type to check
 * @tparam U any number of types to check against
 */
template <class T, class... U>
concept IsNotAnyOf = !IsAnyOf<T, U...>;

/**
 * Check if the type `T` supports the arithmetic operations `+`, `-`, `*`, `/`.
 * @code
 * template <Arithmetic T>
 * void foo(T a, T b); // a and b can be added, subtracted, multiplied and divided with the corresponding operator
 * @endcode
 * @tparam T type to check
 */
template <class T>
concept Arithmetic = requires(T a, T b) {
  { a + b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { a* b } -> std::convertible_to<T>;
  { a / b } -> std::convertible_to<T>;
};  // NOLINT(readability/braces) per C++ standard concept definition

/**
 * Check if the type `T` supports the arithmetic operations `+`, `-`, `*`, `/`
 * and the comparison operators `<`, `>`, `<=`, `>=`.
 * @code
 * template <Numeric T>
 * void foo(T a); // a can be added, subtracted, multiplied, divided and ordered with the corresponding operator
 * @endcode
 * @tparam T type to check
 */
template <class T>
concept Numeric = std::totally_ordered<T> && Arithmetic<T>;

}  // namespace lucid
