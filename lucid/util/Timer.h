/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Timer class.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <ratio>
#include <type_traits>

namespace lucid {

// Forward declaration
struct user_clock;

/**
 * Simple timer class to evaluate the performance of the software.
 * The timer can be started, paused, and resumed.
 * The elapsed time is returned in seconds or as a duration.
 */
template <typename T>
class TimerBase {
 public:
  using clock = T;
  typedef typename clock::duration duration;
  typedef typename clock::time_point time_point;

  /** @constructor{TimerBase} */
  TimerBase();

  /**
   * Start the timer.
   *
   * The timer is reset to zero.
   */
  void start();

  /**
   * Pause the timer.
   *
   * If the timer is not running, this function does nothing.
   */
  void pause();

  /**
   * Resume the timer.
   *
   * If the timer is not running, this function does nothing.
   */
  void resume();

  /** @checker{running, timer} */
  [[nodiscard]] bool is_running() const;
  /** @getter{duration of elapsed time, timer} */
  [[nodiscard]] duration elapsed() const;
  /** @getter{number elapsed seconds, timer} */
  [[nodiscard]] std::chrono::duration<double>::rep seconds() const;

  TimerBase<T> &operator+=(const TimerBase<T> &other);
  TimerBase<T> operator+(const TimerBase<T> &other) const;

 protected:
  /** @getter{current instant, timer} */
  [[nodiscard]] time_point now() const { return clock::now(); }

 private:
  bool running_{false};      ///< Whether the timer is running or not.
  time_point last_start_{};  ///< Last time_point when the timer is started or resumed.
  duration elapsed_{};       ///< Elapsed time so far. This doesn't include the current fragment if it is running.
};

template <typename T>
TimerBase<T> &TimerBase<T>::operator+=(const TimerBase<T> &other) {
  elapsed_ += other.elapsed();
  return *this;
}

template <typename T>
TimerBase<T> TimerBase<T>::operator+(const TimerBase<T> &other) const {
  TimerBase<T> result = *this;
  result += other;
  return result;
}

// Use high_resolution clock if it's steady, otherwise use steady_clock.
using chosen_steady_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                               std::chrono::high_resolution_clock, std::chrono::steady_clock>;

extern template class TimerBase<chosen_steady_clock>;
/** Timer class using the a steady clock. */
class Timer : public TimerBase<chosen_steady_clock> {};

/**
 * Structure that will hold the user clock data.
 * It implements the Clock interface of std::chrono.
 */
struct user_clock {
  typedef uint64_t rep;
  typedef std::micro period;
  typedef std::chrono::duration<rep, period> duration;
  typedef std::chrono::time_point<user_clock> time_point;
  const bool is_steady = false;  // Not sure how this should be interpreted here
  static time_point now();
};

extern template class TimerBase<user_clock>;
/** Timer class using the user_clock. */
class UserTimer : public TimerBase<user_clock> {};

/**
 * The TimeGuard wraps a timer object and pauses it when the guard object is destructed.
 * Useful for measuring the exact time spent in a block of code.
 * @code
 * // Example usage
 * class MyClass {
 *  private:
 *   Timer timer_;
 *  public:
 *   void function_to_measure() {
 *    TimerGuard guard(&timer_, true);
 *    // Code to measure
 *   }
 *   // Return the total time elapsed, even across multiple calls
 *   double time_elapsed() { return timer_.seconds(); }
 * };
 * @endcode
 */
template <class T>
class TimerGuard {
 public:
  /**
   * Construct a new TimeGuard object.
   *
   * If `timer` is a nullptr, the guard will not do anything.
   * If `start_timer` is true, the `timer` starts as soon as the guard is constructed.
   * Otherwise, the method @ref resume has to be called manually to start the timer.
   * @param timer a pointer to the timer object to be guarded. Must remain valid for the lifetime of the guard
   * @param start_timer whether the timer should be started as soon as the guard is created
   */
  explicit TimerGuard(T *timer, bool start_timer = true);

  TimerGuard(const TimerGuard &) = delete;
  TimerGuard(TimerGuard &&) = delete;
  TimerGuard &operator=(const TimerGuard &) = delete;
  TimerGuard &operator=(TimerGuard &&) = delete;

  /**
   * When the timer guard object is destructed, it pauses the embedded timer object.
   * If the timer is a nullptr, this does nothing.
   */
  ~TimerGuard();

  /** Pause the guarded timer object if the timer is not a nullptr */
  void pause();

  /** Resume the guarded timer object if the timer is not a nullptr */
  void resume();

 private:
  T *const timer_;  ///< The timer to be guarded.
};

}  // namespace lucid
