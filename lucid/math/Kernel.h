/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Kernel class.
 */
#pragma once

#include <iosfwd>
#include <memory>
#include <utility>

#include "lucid/lib/eigen.h"
#include "lucid/math/KernelParameter.h"
#include "lucid/util/concept.h"

namespace lucid {

/**
 * Represents a kernel function.
 */
class Kernel {
 public:
  Kernel() = default;
  Kernel(const Kernel&) = default;
  Kernel(Kernel&&) = default;
  Kernel& operator=(const Kernel&) = default;
  Kernel& operator=(Kernel&&) = default;
  virtual ~Kernel() = default;

  /** @checker{is stationary, kernel} */
  [[nodiscard]] virtual bool is_stationary() const = 0;

  /**
   * Retrieves the value of the specified kernel parameter.
   * @tparam T type of the value to retrieve
   * @param parameter kernel parameter to retrieve
   * @return value of the specified kernel parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this kernel
   */
  template <IsAnyOf<int, double, const Vector&> T>
  [[nodiscard]] T get_parameter(KernelParameter parameter) const;

  /**
   * Set an integer parameter for the kernel.
   * @param parameter Specifies the kernel parameter to set
   * @param value integer value to assign to the specified parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this kernel
   */
  virtual void set_parameter(KernelParameter parameter, int value);
  /**
   * Set the specified kernel parameter to a double value.
   * @param parameter The kernel parameter to be set
   * @param value double value to assign to the specified parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this kernel
   */
  virtual void set_parameter(KernelParameter parameter, double value);
  /**
   * Set the specified parameter of the kernel to the provided vector.
   * @param parameter kernel parameter to be set or modified
   * @param value vector value to be assigned to the specified kernel parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this kernel
   */
  virtual void set_parameter(KernelParameter parameter, Vector value);

  /**
   * Compute the kernel function on `x1` and `x2`.
   * @f[
   * K(x_1, x_2)
   * @f]
   * @param x1 first input vector
   * @param x2 second input vector
   * @return kernel value
   */
  virtual Scalar operator()(const Vector& x1, const Vector& x2) const = 0;
  /**
   * Compute the kernel function on `x`.
   * @f[
   * K(x, x)
   * @f]
   * @param x input vector
   * @return kernel value
   */
  Scalar operator()(const Vector& x) const { return (*this)(x, x); }

  /**
   * Clone the kernel.
   * Create a new instance of the kernel with the same parameters.
   * @return new instance of the kernel
   */
  [[nodiscard]] virtual std::unique_ptr<Kernel> clone() const = 0;

 protected:
  /**
   * Retrieves the value of the specified kernel parameter.
   * @param parameter kernel parameter to retrieve
   * @return value of the specified kernel parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this kernel
   */
  [[nodiscard]] virtual int get_parameter_i(KernelParameter parameter) const;
  /**
   * Retrieves the value of the specified kernel parameter.
   * @param parameter kernel parameter to retrieve
   * @return value of the specified kernel parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this kernel
   */
  [[nodiscard]] virtual double get_parameter_d(KernelParameter parameter) const;
  /**
   * Retrieves the value of the specified kernel parameter.
   * @param parameter kernel parameter to retrieve
   * @return value of the specified kernel parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this kernel
   */
  [[nodiscard]] virtual const Vector& get_parameter_v(KernelParameter parameter) const;
};

std::ostream& operator<<(std::ostream& os, const Kernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Kernel)

#endif
