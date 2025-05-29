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
#include "lucid/model/Parameter.h"
#include "lucid/model/Parametrizable.h"

namespace lucid {

/**
 * Represents a kernel function.
 */
class Kernel : public Parametrizable {
 public:
  Kernel() = default;

  /** @checker{is stationary, kernel} */
  [[nodiscard]] virtual bool is_stationary() const = 0;
  /** @checker{is isotropic, kernel} */
  [[nodiscard]] virtual bool is_isotropic() const = 0;

  /**
   * Compute the kernel function on `x` and `y`.
   * @f[
   * K(x, y)
   * @f]
   * @tparam DerivedX type of the first input matrix
   * @tparam DerivedY type of the second input matrix
   * @param x @nxd first input row matrix
   * @param y @nxd second input row matrix
   * @return kernel value
   */
  template <class DerivedX, class DerivedY>
  Matrix operator()(const MatrixBase<DerivedX>& x, const MatrixBase<DerivedY>& y) const {
    return (*this)(x, y, nullptr);
  }
  /**
   * Compute the kernel function on `x`.
   * @f[
   * K(x, x)
   * @f]
   * @tparam Derived type of the input matrix
   * @param x @nxd input matrix
   * @return kernel value
   */
  template <class Derived>
  Matrix operator()(const MatrixBase<Derived>& x) const {
    const Eigen::Ref<const Matrix> x_ref{x};
    return (*this)(x_ref, x_ref, nullptr);
  }
  /**
   * Compute the kernel function on `x`.
   * @f[
   * K(x, x)
   * @f]
   * Moreover, compute the gradient of the kernel function and store it in `gradient`.
   * @tparam Derived type of the input matrix
   * @param x @nxd input matrix
   * @param[out] gradient gradient of the kernel function with respect to the parameters
   * @return kernel value
   */
  template <class Derived>
  Matrix operator()(const MatrixBase<Derived>& x, double& gradient) const {
    const Eigen::Ref<const Matrix> x_ref{x};
    return (*this)(x_ref, x_ref, &gradient);
  }

  /**
   * Clone the kernel.
   * Create a new instance of the kernel with the same parameters.
   * @return new instance of the kernel
   */
  [[nodiscard]] virtual std::unique_ptr<Kernel> clone() const = 0;

 protected:
  /**
   * Compute the kernel function on `X1` and `X2`.
   * @f[
   * K(X_1, X_2)
   * @f]
   * If `gradient` is not `nullptr`, the gradient of the kernel function with respect to the parameters
   * is computed and stored in `*gradient`.
   * @param X1 @nxdx first input matrix
   * @param X2 @nxdy second input matrix
   * @param[out] gradient pointer to store the gradient of the kernel function with respect to the parameters
   * @return kernel value
   */
  virtual Matrix operator()(ConstMatrixRef X1, ConstMatrixRef X2, double* gradient) const = 0;
};

std::ostream& operator<<(std::ostream& os, const Kernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Kernel)

#endif
