/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Set class.
 */
#pragma once

#include <iosfwd>
#include <string>

#include "lucid/lib/eigen.h"

namespace lucid {

/**
 * Generic set over a @d dimensional vector space @XsubRd.
 * It can be used to test if a vector is in the set and to sample elements from the set.
 */
class Set {
 public:
  Set() = default;
  Set(const Set&) = default;
  Set(Set&&) = default;
  Set& operator=(const Set&) = default;
  Set& operator=(Set&&) = default;
  virtual ~Set() = default;

  /** @getter{dimension, set @X} */
  [[nodiscard]] virtual Dimension dimension() const = 0;
  /**
   * Extract @N elements from @X using some kind of random distribution.
   * @pre `num_samples` must be greater than 0
   * @note The seed for the random number generator can be set using @ref random::seed.
   * @param num_samples number of samples to generate @N
   * @return @nxd matrix of samples, where @d is the dimension of @X.
   * In other words, the samples are stored as rows vectors in the matrix
   */
  [[nodiscard]] virtual Matrix sample(Index num_samples) const = 0;
  /**
   * Extract an element from @X using some kind of random distribution.
   * @note The seed for the random number generator can be set using @ref random::seed.
   * @return element of the set
   */
  [[nodiscard]] Vector sample() const;
  /**
   * Check if a vector is in @X.
   * @pre @x must have the same dimension as the set
   * @param x vector to test
   * @return true if @x is in the set
   * @return false if @x is not in the set
   */
  [[nodiscard]] bool contains(ConstVectorRef x) const { return (*this)(x); }

  /**
   * Check if a vector is in @X.
   * @pre @x must have the same dimension as the set
   * @param x vector to test
   * @return true if @x is in the set
   * @return false if @x is not in the set
   */
  [[nodiscard]] virtual bool operator()(ConstVectorRef x) const = 0;

  /**
   * Change the size of the set.
   * The size change can be different for each dimension.
   * For example, for a rectangular set, this would change the lower and upper bounds so that the original set sits in
   * the center of the new set, which has its size changed by the specified amounts.
   * @code{.unparsed}
   * ┌───────────┐
   * │    ┌─┐    │
   * │    └─┘    │
   * └───────────┘
   * @endcode
   * @pre The set must support size changes.
   * @pre The new size must be non-negative in all dimensions.
   * @param delta_size amount to change the size of the set
   */
  void change_size(double delta_size);

  /**
   * Change the size of the set.
   * The size change can be different for each dimension.
   * For example, for a rectangular set, this would change the lower and upper bounds so that the original set sits in
   * the center of the new set, which has its size changed by the specified amounts.
   * @code{.unparsed}
   * ┌───────────┐
   * │    ┌─┐    │
   * │    └─┘    │
   * └───────────┘
   * @endcode
   * @pre The set must support size changes.
   * @pre The size of `delta_size` must be equal to the dimension of the set.
   * @pre The new size must be non-negative in all dimensions.
   * @param delta_size vector of amounts to change the size of the set for each dimension
   */
  virtual void change_size(ConstVectorRef delta_size);

  /**
   * Generate a lattice of points in the set.
   * @param points_per_dim number of points per each dimension
   * @param endpoint whether to include the endpoints of the lattice
   * @return lattice of points in the set
   */
  [[nodiscard]] Matrix lattice(Index points_per_dim, bool endpoint = false) const;
  /**
   * Generate a lattice of points in the set.
   * @param points_per_dim number of points per each dimension
   * @param endpoint whether to include the endpoints of the lattice
   * @return lattice of points in the set
   */
  [[nodiscard]] virtual Matrix lattice(const VectorI& points_per_dim, bool endpoint) const = 0;

  /**
   * Extract @N elements element from @X using some kind of random distribution, where @N is the number of rows in @x.
   * @param[out] x @nxd matrix to store the elements
   * @return @nxd matrix of samples, where @d is the dimension of @X.
   */
  template <class Derived>
  Eigen::MatrixBase<Derived>& operator>>(Eigen::MatrixBase<Derived>& x) const {
    if constexpr (Derived::ColsAtCompileTime != 1) {
      return x = sample(x.rows());
    }
    return x = sample().transpose();
  }
};

std::ostream& operator<<(std::ostream& os, const Set& set);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Set)

#endif
