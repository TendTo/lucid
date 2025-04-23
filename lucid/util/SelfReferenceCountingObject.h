/**
 * @author Ernesto Casablanca (casablancaernesto@gmail.com)
 * @copyright 2024 lucid
 * @licence BSD 3-Clause License
 * SelfReferenceCountingObject class.
 */
#pragma once

#include <cstddef>

#ifdef LUCID_THREAD_SAFE
#include <atomic>
#endif

namespace lucid {

/**
 * Utility class to be inherited from to obtain compatibility with the intrusive_ptr.
 * It implements both @ref AddRef and @ref Release methods and supports both thread safe and unsafe reference counting.
 * This class (and its subclasses) are meant to be initialised and used through an intrusive_ptr.
 * @code
 * class MyClass : public SelfReferenceCountingObject {
 *  public:
 *    static intrusive_ptr New() { return intrusive_ptr new MyClass(); }
 *  private:
 *    MyClass() = default;
 * };
 *
 * int main() {
 *   const intrusive_ptr<MyClass> ptr{MyClass::New()};
 * }
 * @endcode
 */
class SelfReferenceCountingObject {
 public:
  /** @addref{object} */
  void add_ref() {
#ifdef LUCID_THREAD_SAFE
    ref_count_.fetch_add(1);
#else
    ref_count_++;
#endif
  }
  /** @release{object} */
  void release() {
#ifdef LUCID_THREAD_SAFE
    if (ref_count_.fetch_sub(1) == 1) {
#else
    if (--ref_count_ == 0) {
#endif
      delete this;
    }
  }

  /** @getter{reference counter, object} */
  [[nodiscard]] std::size_t use_count() const noexcept {
#ifdef LUCID_THREAD_SAFE
    return ref_count_.load();
#else
    return ref_count_;
#endif
  }

 protected:
  /**
   * Virtual destructor for the SelfReferenceCountingObject.
   * Needed to ensure that the derived classes are deleted correctly.
   */
  virtual ~SelfReferenceCountingObject() = default;

 private:
#ifdef LUCID_THREAD_SAFE
  std::atomic<std::size_t> ref_count_;  ///< Thread safe reference counter
#else
  std::size_t ref_count_{0};  ///< Reference counter
#endif
};  // NOLINT(readability/braces) per C++ standard concept definition

}  // namespace lucid
