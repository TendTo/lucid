/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * ScopedValue class.
 */
#pragma once

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace lucid {

// Forward declaration
template <class T, class... Tags>
class ScopedValueShield;

/**
 * Base class for creating a stack of scoped values of type T, identified by Tags.
 * This class maintains a linked vector of instances, allowing easy access to the top (most recently created)
 * and bottom (first created) instances.
 * @code
 * #include <iostream>
 * #include "lucid/util/ScopedValue.h"
 *
 * struct Config {
 *   std::string path;
 *   int level = 0;
 * };
 *
 * const Config default_config{"default_path", 0};
 *
 * // The tag avoids conflicts with other ScopedValue<Config> usages.
 * using ScopedConfig = lucid::ScopedValue<Config, struct CommandLineConfigTag>;
 *
 * void fun() {  // Note there are no parameters. This could be at any level of the call stack.
 *   // If there is any scoped config on the stack, use it. Otherwise, use the default config.
 *   const Config& config = ScopedConfig::top() ? ScopedConfig::top()->value() : default_config;
 *   std::cout << "Path: " << config.path << ", level: " << config.level << std::endl;
 * }
 *
 * int main() {
 *   fun();  // Path: default_path, level: 0
 *   {
 *     ScopedConfig config1{"path 1", 1};  // Push new config onto the stack
 *     fun();                              // Path: path 1, level: 1
 *     {
 *       fun();                              // Path: path 1, level: 1
 *       ScopedConfig config2{"path 2", 2};  // Push another config onto the stack
 *       fun();                              // Path: path 2, level: 2
 *     }  // config2 goes out of scope and is popped from the stack
 *     fun();  // Path: path 1, level: 1
 *   }  // config1 goes out of scope and is popped from the stack
 *   fun();  // Path: default_path, level: 0
 * }
 * @endcode
 * @tparam T Type of the value to be scoped.
 * @tparam Tags Variadic template parameters to uniquely identify different scoped value types.
 */
template <class T, class... Tags>
class BaseScopedValue {
 public:
  /** Construct a new scoped value instance and add it to the scope stack of instances. */
  BaseScopedValue() { scope_stack_.emplace_back(this); }

  /**
   * Copy constructor that adds the current instance to the scope stack of instances.
   * The value is copied from the other instance.
   */
  BaseScopedValue(const BaseScopedValue&) { scope_stack_.emplace_back(this); }
  BaseScopedValue& operator=(const BaseScopedValue& other) = default;

  BaseScopedValue(BaseScopedValue&& other) = delete;
  BaseScopedValue& operator=(BaseScopedValue&& other) = delete;

  /** Destructor that removes the current instance from the scope stack of instances. */
  virtual ~BaseScopedValue() { detach(); }

  /** @getsetter{value, current scoped value} */
  T& value() { return const_cast<T&>(static_cast<const BaseScopedValue*>(this)->value()); }
  /** @getter{value, current scoped value} */
  virtual const T& value() const = 0;

  T& operator*() { return value(); }
  const T& operator*() const { return value(); }
  T* operator->() { return &value(); }
  const T* operator->() const { return &value(); }

  /**
   * Swap the values of two scoped value instances.
   * Both instances must be of the same type and identified by the same Tags.
   * Their order in the vector of instances remains unchanged.
   * @param other the other scoped value instance to swap with
   */
  void swap(BaseScopedValue& other) noexcept { std::swap(value(), other.value()); }

  /** @getter{most recently created scoped value instance, scope stack} */
  static BaseScopedValue* top() { return scope_stack_.empty() ? nullptr : scope_stack_.back(); }
  /** @getter{oldest created scoped value instance, scope stack} */
  static BaseScopedValue* bottom() { return scope_stack_.empty() ? nullptr : scope_stack_.front(); }
  /** @getter{all scoped value instances, scope stack} */
  static const std::vector<BaseScopedValue*>& scope_stack() { return scope_stack_; }
  /**
   * Set the entire stack of scoped value instances.
   * Useful for initialising the scope stack in other threads.
   * @note The entire current stack will be replaced.
   * @param new_scopes new stack of scoped value instances.
   */
  static void set_scopes(std::vector<BaseScopedValue*> new_scopes) { scope_stack_ = std::move(new_scopes); }

  // Disable the use of the default new and delete operators, as scoped instances should not be created on the heap.
  void* operator new(std::size_t) = delete;           ///< Disabled standard new
  void* operator new[](std::size_t) = delete;         ///< Disabled array new
  void* operator new(std::size_t, void*) = delete;    ///< Disabled placement new
  void* operator new[](std::size_t, void*) = delete;  ///< Disabled placement array new

 private:
  /** Remove the current instance from the scope stack. */
  void detach() {
    for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
      if (*it == this) {
        scope_stack_.erase(std::next(it).base());
        return;
      }
    }
  }

  static inline thread_local std::vector<BaseScopedValue*> scope_stack_;  ///< Stack of all scoped value instances.

  friend ScopedValueShield<T, Tags...>;
};

/**
 * Generic scoped value class that can hold a value of type T, while interfacing with a base class B.
 * This allows for polymorphic behavior when T is derived from B.
 * The class maintains a stack of instances, allowing access to the most recently created (top)
 * and the first created (bottom) instances.
 * @tparam T Type of the value to be scoped. Must be derived from B.
 * @tparam B Base class type for polymorphic behavior.
 * @tparam Tags Variadic template parameters to uniquely identify different scoped value types.
 */
template <class T, class B, class... Tags>
class PolymorphicScopedValue final : public BaseScopedValue<B, Tags...> {
 public:
  using BaseScopedValue<B, Tags...>::value;

  template <class... Args>
    requires std::is_constructible_v<T, Args...>
  explicit PolymorphicScopedValue(Args&&... args)
      : BaseScopedValue<B, Tags...>(), value_{std::forward<Args>(args)...} {}
  PolymorphicScopedValue(const PolymorphicScopedValue& other) = default;
  PolymorphicScopedValue(PolymorphicScopedValue&& other) = default;
  ~PolymorphicScopedValue() override = default;
  PolymorphicScopedValue& operator=(const PolymorphicScopedValue& other) = default;
  PolymorphicScopedValue& operator=(PolymorphicScopedValue&& other) = default;

  const B& value() const override { return value_; }

 private:
  T value_;
};

/**
 * Simplified scoped value class that holds a value of type T.
 * This class is a specialization of PolymorphicScopedValue where T and B are the same type.
 * It maintains a stack of instances, allowing access to the most recently created (top)
 * and the first created (bottom) instances.
 * @tparam T Type of the value to be scoped.
 * @tparam Tags Variadic template parameters to uniquely identify different scoped value types.
 */
template <class T, class... Tags>
using ScopedValue = PolymorphicScopedValue<T, T, Tags...>;

/**
 * A utility class that temporarily shields all current scoped values of type T and Tags.
 * When an instance of this class is created, it saves the current stack of scoped values
 * and clears the stack, effectively "shielding" any existing scoped values.
 * When the instance is destroyed, it restores the saved stack of scoped values.
 * @tparam T Type of the value to be scoped.
 * @tparam Tags Variadic template parameters to uniquely identify different scoped value types.
 */
template <class T, class... Tags>
class ScopedValueShield {
  using SavedScopedValue = BaseScopedValue<T, Tags...>;

 public:
  /** Create a new shield instance, saving and clearing the current stack of scoped values. */
  ScopedValueShield() : saved_scope_stack_{SavedScopedValue::scope_stack_} { SavedScopedValue::scope_stack_.clear(); }
  /** Restore the saved stack of scoped values upon destruction. */
  ~ScopedValueShield() { SavedScopedValue::scope_stack_ = std::move(saved_scope_stack_); }

 private:
  std::vector<SavedScopedValue*> saved_scope_stack_;  ///< Saved stack of scoped value instances.
};

}  // namespace lucid
