/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/util/Timer.h"

#include <sys/resource.h>

#include <stdexcept>

#include "lucid/util/exception.h"
#include "lucid/util/logging.h"

namespace lucid {

template <class T>
TimerBase<T>::TimerBase() : last_start_{now()} {}

template <class T>
void TimerBase<T>::start() {
  LUCID_TRACE("TimerBase::Start");
  last_start_ = now();
  elapsed_ = duration{0};
  running_ = true;
}

template <class T>
void TimerBase<T>::pause() {
  if (running_) {
    running_ = false;
    elapsed_ += (now() - last_start_);
  }
}

template <class T>
void TimerBase<T>::resume() {
  if (!running_) {
    last_start_ = now();
    running_ = true;
  }
}

template <class T>
bool TimerBase<T>::is_running() const {
  return running_;
}

template <class T>
typename TimerBase<T>::duration TimerBase<T>::elapsed() const {
  LUCID_TRACE("TimerBase::duration");
  return running_ ? elapsed_ + (now() - last_start_) : elapsed_;
}

template <class T>
std::chrono::duration<double>::rep TimerBase<T>::seconds() const {
  LUCID_TRACE("TimerBase::seconds");
  return std::chrono::duration_cast<std::chrono::duration<double>>(elapsed()).count();
}

user_clock::time_point user_clock::now() {
  LUCID_TRACE("user_clock::now");
  struct rusage usage{};
  if (0 != getrusage(RUSAGE_SELF, &usage))
    throw exception::LucidException("Failed to get current resource usage (getrusage)");
  return time_point(duration(static_cast<uint64_t>(usage.ru_utime.tv_sec) * std::micro::den +
                             static_cast<uint64_t>(usage.ru_utime.tv_usec)));
}

// Explicit instantiations
template class TimerBase<chosen_steady_clock>;
template class TimerBase<user_clock>;

TimerGuard::TimerGuard(Timer *const timer, const bool start_timer) : timer_{timer} {
  if (start_timer && timer_) timer_->resume();
}

TimerGuard::~TimerGuard() {
  if (timer_) timer_->pause();
}

void TimerGuard::pause() {
  if (timer_) timer_->pause();
}

void TimerGuard::resume() {
  if (timer_) timer_->resume();
}

}  // namespace lucid