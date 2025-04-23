/**
 * @author Ernesto Casablanca (casablancaernesto@gmail.com)
 * @copyright 2024 delpi
 * @copyright Electronic Arts Inc. All rights reserved.
 * @licence BSD 3-Clause License
 * intrusive_ptr class.
 */
#pragma once

#include <utility>

namespace lucid {

/**
 * Pointer to a generic object that supports intrusive reference counting.
 * Implementation based on the
 * [EASTL intrusive_ptr](https://github.com/electronicarts/EASTL/blob/master/include/EASTL/intrusive_ptr.h).
 * @tparam T class the intrusive_ptr is pointing to.
 * Must have `add_ref` and `release` methods.
 */
template <class T>
class intrusive_ptr {
 public:
  typedef T element_type;  ///< The type this intrusive_ptr points to.

  /** @constructor{intrusive_ptr} */
  intrusive_ptr() : ptr_{nullptr} {}

  /**
   * Construct a new intrusive_ptr object wrapping the given pointer `p`.
   * If `add_ref` is true, the reference count of the object is incremented immediately.
   * @code
   * intrusive_ptr<MyObject> str(new MyObject());
   * @endcode
   * @param p pointer to the object
   * @param add_ref whether to increment the reference count
   */
  explicit intrusive_ptr(T* p, const bool add_ref = true) : ptr_{p} {
    if (ptr_ && add_ref) ptr_->add_ref();
  }
  /**
   * Construct a new intrusive_ptr object from another intrusive_ptr object.
   * `add_ref` is immediately run on the object.
   * The source intrusive_ptr object maintains its `add_ref` on the pointer.
   * @code
   * intrusive_ptr<MyObject> my_object_ptr_1;
   * intrusive_ptr<MyObject> my_object_ptr_2(my_object_ptr_1);
   * @endcode
   * @param ip intrusive_ptr object to copy
   */
  intrusive_ptr(const intrusive_ptr& ip) : ptr_{ip.ptr_} {
    if (ptr_) ptr_->add_ref();
  }
  /**
   * Construct a new intrusive_ptr object from another intrusive_ptr object using move semantics.
   * @param ip intrusive_ptr object to move
   */
  intrusive_ptr(intrusive_ptr&& ip) noexcept : ptr_{ip.ptr_} { ip.ptr_ = nullptr; }

  /** Destruct the intrusive_ptr object, releasing the owned pointer. */
  ~intrusive_ptr() {
    if (ptr_) ptr_->release();
  }

  /**
   * Copy assignment operator.
   * `add_ref` is immediately run on the object.
   * The source `ip` maintains its reference to the pointer.
   * If this pointer was already in use, the old object is released at the end.
   * @param ip intrusive_ptr object to copy
   * @return reference to this intrusive_ptr object
   */
  intrusive_ptr& operator=(const intrusive_ptr& ip) {
    intrusive_ptr{ip}.swap(*this);
    return *this;
  }

  /**
   * Move assignment operator.
   * @param ip intrusive_ptr object to move
   * @return reference to this intrusive_ptr object
   */
  intrusive_ptr& operator=(intrusive_ptr&& ip) noexcept {
    // NOLINTNEXTLINE(runtime/explicit) per C++ standard concept definition
    intrusive_ptr(static_cast<intrusive_ptr&&>(ip)).swap(*this);
    return *this;
  }

  /**
   * Copy assignment operator.
   * `add_ref` is immediately run on the object.
   * If this pointer was already in use, the old object is released at the end.
   * @param ptr raw pointer to the object
   * @return reference to this intrusive_ptr object
   */
  intrusive_ptr& operator=(T* ptr) {
    intrusive_ptr{ptr}.swap(*this);
    return *this;
  }

  /** @getter{reference to the contained object, intrusive_ptr} */
  T& operator*() const { return *ptr_; }
  /**
   * @getter{underlying raw pointer, intrusive_ptr,
   * Allows this object to be used as if it were the underlying pointer.}
   */
  T* operator->() const { return ptr_; }
  /** @getter{underlying raw pointer, intrusive_ptr} */
  [[nodiscard]] T* get() const { return ptr_; }

  /** Release the owned pointer, decrementing the reference count and setting @ref ptr_ to null. */
  void reset() { intrusive_ptr{}.swap(*this); }

  /**
   * Exchanges the owned pointer between two intrusive_ptr objects.
   * @note The reference count is not modified.
   * @param ip intrusive_ptr object to swap with
   */
  void swap(intrusive_ptr& ip) noexcept { std::swap(ptr_, ip.ptr_); }

  /**
   * Sets the owned pointer to the given pointer `ptr` without incrementing the reference count.
   * The intrusive_ptr eventually only does a `release()` on the object.
   * Useful for assuming a reference that someone else has handed you and making sure it is always released.
   * @param ptr pointer to the object
   */
  void attach(T* ptr) {
    T* const temp_ptr = ptr_;
    ptr_ = ptr;
    if (temp_ptr) temp_ptr->release();
  }

  /**
   * Surrenders the reference held by an intrusive_ptr pointer.
   * It returns the current reference and nulls the pointer without decrementing the reference count.
   * Therefore, if the returned pointer is non-null it must be eventually released by the caller.
   * @return the current reference
   */
  T* detach() {
    T* const pTemp = ptr_;
    ptr_ = nullptr;
    return pTemp;
  }

  typedef element_type* intrusive_ptr::* bool_;
  /**
   * Implicit bool conversion.
   * Allows the intrusive_ptr to be used as a boolean in checks.
   * @code
   * intrusive_ptr<MyObject> ptr = new MyObject();
   * if (ptr) ptr->DoSomething();
   * @endcode
   * @note The implementation does not use the builtin `bool()` operator.
   * That is because boolean are implicitly converted to short, int, float, etc.
   * Hence, you can have ambiguous code such as `if (ptr == 1) // True`.
   */
  operator bool_() const noexcept { return ptr_ ? &intrusive_ptr::ptr_ : nullptr; }
  bool operator!() const { return ptr_ == nullptr; }

 protected:
  T* ptr_;  ///< Raw pointer to the data
};

}  // namespace lucid
