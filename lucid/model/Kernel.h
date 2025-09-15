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
#include <vector>

#include "lucid/lib/eigen.h"
#include "lucid/model/Parameter.h"
#include "lucid/model/Parametrizable.h"

namespace lucid {

/**
 * Represents a kernel function.
 * Given a vector space @XsubRd, a kernel @f$ k : \mathcal{X} \times \mathcal{X} \to \mathbb{R} @f$
 * is a positive definite function that uniquely identifies a reproducing kernel Hilbert space (RKHS) @H
 * that contains functions @f$ f : \mathcal{X} \to \mathbb{R} @f$.
 * Moreover, we are guaranteed the property
 * @f[
 * f(x) = \langle f, k(x, \cdot) \rangle_\mathcal{H}
 * @f]
 * for all @f$ x \in \mathcal{X} @f$ and @f$ f \in \mathcal{H} @f$,
 */
class Kernel : public Parametrizable {
 public:
  explicit Kernel(const Parameters parameters = NoParameters) : Parametrizable{parameters} {}

  /** @checker{is stationary, kernel} */
  [[nodiscard]] virtual bool is_stationary() const = 0;

  /**
   * Compute the kernel function on @x1 and @x2, both being matrices of row vectors in @XsubRd,
   * @f[
   * k(x_1, x_2) .
   * @f]
   * @pre `x1` and `x2` must have the same number of columns.
   * @tparam DerivedX type of the first input matrix
   * @tparam DerivedY type of the second input matrix
   * @param x1 @n1xd first input row matrix
   * @param x2 @n2xd second input row matrix
   * @return kernel value
   */
  template <class DerivedX, class DerivedY>
  Matrix operator()(const MatrixBase<DerivedX>& x1, const MatrixBase<DerivedY>& x2) const {
    return (*this)(x1, x2, nullptr);
  }
  /**
   * Compute the kernel function on @x, which is a matrix of row vectors in @XsubRd,
   * @f[
   * k(x, x) .
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
   * Compute the kernel function on @x, which is a matrix of row vectors in @XsubRd,
   * @f[
   * k(x, x) .
   * @f]
   * Moreover, compute the gradient of the kernel function and store it in `gradient`.
   * @tparam Derived type of the input matrix
   * @param x @nxd input matrix
   * @param[out] gradient gradient of the kernel function with respect to the parameters dimensions
   * @return kernel value
   */
  template <class Derived>
  Matrix operator()(const MatrixBase<Derived>& x, std::vector<Matrix>& gradient) const {
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
   * Compute the kernel function on @x1 and @x2, both being matrices of row vectors in @XsubRd,
   * @f[
   * k(x_1, x_2) .
   * @f]
   * If `gradient` is not `nullptr`, the gradient of the kernel function with respect to the parameters
   * is computed and stored in `*gradient`.
   * @pre `x1` and `x2` must have the same number of columns.
   * @param x1 @n1xd first input matrix
   * @param x2 @n2xd second input matrix
   * @param[out] gradient pointer to store the gradient of the kernel function with respect to the kernel parameters
   * @return kernel value
   */
  Matrix operator()(ConstMatrixRef x1, ConstMatrixRef x2, std::vector<Matrix>* gradient) const;
  /**
   * Concrete implementation of @ref operator()().
   * @param x1 @n1xd first input matrix
   * @param x2 @n2xd second input matrix
   * @param[out] gradient pointer to store the gradient of the kernel function with respect to the kernel
   * parameters
   * @return kernel value
   */
  virtual Matrix apply_impl(ConstMatrixRef x1, ConstMatrixRef x2, std::vector<Matrix>* gradient) const = 0;
};

std::ostream& operator<<(std::ostream& os, const Kernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Kernel)

#endif
