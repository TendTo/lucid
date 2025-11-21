/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Set class.
 */
#pragma once

#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lucid/lib/eigen.h"

namespace lucid {

// Forward declaration
class RectSet;

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
   * @note The seed for the random number generator can be set using @ref random::seed.
   * @pre `num_samples` must be greater than 0
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

  /** @overload */
  [[nodiscard]] bool contains_wrapped(ConstVectorRef x, ConstVectorRef period, Dimension num_periods) const;
  /**
   * Check if a vector is in @X, having the vector wrapped around a given period.
   * Given a point `x`, we assume that the space is tiled every `period` units in each dimension,
   * and we check if any of the tiled copies of `x` is in the set.
   * Graphically,
   * @code{.unparsed}
   *        ▲
   *  ┌-----┼-----------┐
   *  |x    │x     x    |
   *  |     │           |
   *  |     ├─────┐     |
   *  |x    │x    │x    |
   * ─┼─────┼─────┴─────┼►
   *  |     │           |
   *  |x    │x     x    |
   *  └-----┼-----------┘
   * @endcode
   * where the continuous rectangular area in the middle represents the periodic space.
   * The original point is indicated by the x in the middle,
   * while all other `xs` represent the additional points that will be checked for membership in the set,
   * given that `num_periods` is set to 1.
   * The wrapping is done by shifting `x` by integer multiples of `period` in each dimension, up to the specified
   * `num_periods`.
   * @note The periodic set starts at the origin.
   * @pre @x must have the same dimension as the set.
   * @pre @x must fall in the range defined by `[0, period)` in all dimensions.
   * @pre `period` must be strictly positive in all dimensions.
   * @pre The set must fit inside the box defined by [-num_periods * period, num_periods * period] in all dimensions.
   * @param x vector to test
   * @param period period for wrapping around
   * @param num_periods number of periods to consider in each direction
   * @return true if any of the wrapped copies of @x is in the set
   * @return false if none of the wrapped copies of @x is in the set
   */
  [[nodiscard]] bool contains_wrapped(ConstVectorRef x, ConstVectorRef period,
                                      const std::vector<Dimension>& num_periods) const;
  /** @overload */
  [[nodiscard]] bool contains_wrapped(ConstVectorRef x, ConstVectorRef period, Dimension num_periods_below,
                                      Dimension num_periods_above) const;
  /**
   * Check if a vector is in @X, having the vector wrapped around a given period.
   * Given a point `x`, we assume that the space is tiled every `period` units in each dimension,
   * and we check if any of the tiled copies of `x` is in the set.
   * Graphically,
   * @code{.unparsed}
   *        ▲
   *  ┼-----------┐
   *  │x     x    |
   *  │           |
   *  ├─────┐     |
   *  │x    │x    |
   * ─┼─────┴─────┼►
   * @endcode
   * where the continuous rectangular area in the middle represents the periodic space.
   * The original point is indicated by the x in the middle,
   * while all other `xs` represent the additional points that will be checked for membership in the set,
   * given that `num_periods_below` is set to 0 and `num_periods_above` is set to 1 (for all dimensions).
   * The wrapping is done by shifting `x` by integer multiples of `period` in each dimension, up to the specified
   * `num_periods`.
   * @note The periodic set starts at the origin.
   * @pre @x must have the same dimension as the set.
   * @pre @x must fall in the range defined by `[0, period)` in all dimensions.
   * @pre `period` must be strictly positive in all dimensions.
   * @pre `num_periods_below` and `num_periods_above` must have the same size as the dimension of the set.
   * @pre `num_periods_below` and `num_periods_above` must be non-negative in all dimensions.
   * @pre The set must fit inside the box defined by [-num_periods_below * period, num_periods_above * period]
   * in all dimensions.
   * @param x vector to test
   * @param period period for wrapping around
   * @param num_periods_below number of periods below the original periodic set to consider in each direction
   * @param num_periods_above number of periods above the original periodic set to consider in each direction
   * @return true if any of the wrapped copies of @x is in the set
   * @return false if none of the wrapped copies of @x is in the set
   */
  [[nodiscard]] bool contains_wrapped(ConstVectorRef x, ConstVectorRef period,
                                      const std::vector<Dimension>& num_periods_below,
                                      const std::vector<Dimension>& num_periods_above) const;

  /**
   * Check if a vector is in @X, having the vector wrapped around a given period.
   * Given a point `x`, we assume that the space is tiled every `period` units in each dimension,
   * and we check if any of the tiled copies of `x` is in the set.
   * Graphically,
   * @code{.unparsed}
   *        ▲
   *  ┌-----┼-----------┐
   *  |x    │x     x    |
   *  |     │           |
   *  |     ├─────┐     |
   *  |x    │x    │x    |
   * ─┼─────┼─────┴─────┼►
   *  |     │           |
   *  |x    │x     x    |
   *  └-----┼-----------┘
   * @endcode
   * where the continuous rectangular area in the middle represents the periodic space.
   * The original point is indicated by the x in the middle,
   * while all other `xs` represent the additional points that will be checked for membership in the set.
   * The number of `xs` in each direction is determined using @ref general_lower_bound and @ref general_upper_bound
   * with respect to the given `period`.
   * @note The periodic set starts at the origin.
   * @pre @x must have the same dimension as the set
   * @pre @x must fall in the range defined by `[0, period)` in all dimensions
   * @pre `period` must be strictly positive in all dimensions.
   * @param x vector to test
   * @param period period for wrapping around
   * @return true if any of the wrapped copies of @x is in the set
   * @return false if none of the wrapped copies of @x is in the set
   */
  [[nodiscard]] bool contains_wrapped(ConstVectorRef x, ConstVectorRef period) const;

  /**
   * Filter a set `xs`, returning only the row vectors that are in @X.
   * @pre `xs` must have the same number of columns as the dimension of the set, @d
   * @param xs @nxd matrix of row vectors to filter
   * @return matrix of row vectors that are in the set
   */
  [[nodiscard]] Matrix include(ConstMatrixRef xs) const;

  /**
   * Filter a set `xs`, returning a mask containing the indices corresponding to the row vectors that are in @X.
   * @pre `xs` must have the same number of columns as the dimension of the set, @d
   * @param xs @nxd matrix of row vectors to filter
   * @return vector of indices corresponding to the vectors that are in the set
   */
  [[nodiscard]] std::vector<Index> include_mask(ConstMatrixRef xs) const;

  /**
   * Filter `xs`, return only the row vectors that are NOT in @X
   * @pre `xs` must have the same number of columns as the dimension of the set, @d
   * @param xs @nxd matrix of row vectors to filter
   * @return matrix of row vectors that are NOT in the set
   */
  [[nodiscard]] Matrix exclude(ConstMatrixRef xs) const;

  /**
   * Filter a set `xs`, returning a mask containing the indices corresponding to the row vectors that are NOT in @X.
   * @pre `xs` must have the same number of columns as the dimension of the set, @d
   * @param xs @nxd matrix of row vectors to filter
   * @return vector of indices corresponding to the vectors that are NOT in the set
   */
  [[nodiscard]] std::vector<Index> exclude_mask(ConstMatrixRef xs) const;

  /**
   * Filter a set `xs`, returning masks containing the indices corresponding to the row vectors that are in @X and
   * NOT in @X.
   * The union of the two sets of indices covers all the indices of `xs`.
   * @pre `xs` must have the same number of columns as the dimension of the set, @d
   * @param xs @nxd matrix of row vectors to filter
   * @return pair of vectors of indices where
   * - the first vector contains the indices corresponding to the vectors that are in the set
   * - the second vector contains the indices corresponding to the vectors that are NOT in the set
   */
  [[nodiscard]] std::pair<std::vector<Index>, std::vector<Index>> include_exclude_masks(ConstMatrixRef xs) const;

  /**
   * Check if a vector is in @X.
   * @pre @x must have the same dimension as the set
   * @param x vector to test
   * @return true if @x is in the set
   * @return false if @x is not in the set
   */
  [[nodiscard]] virtual bool operator()(ConstVectorRef x) const = 0;

  /**
   * Scale the set by the given factor while keeping it inside the given bounds.
   * If any dimension exceeds the bounds after scaling, it is wrapped around to the other side,
   * as another set.
   * The scaling is performed with respect to the center of the set.
   * The scaling factor can be computed relative to either
   * - the current size of the set;
   * - the size of the bounding rectangular set.
   * @param scale scaling factor
   * @param bounds bounding rectangular set
   * @param relative_to_bounds if true, the scaling factor is computed relative to the size of the bounding
   * rectangular set; if false, the scaling factor is computed relative to the current size of the rectangular
   * @return new scaled rectangular set
   */
  [[nodiscard]] std::unique_ptr<Set> scale_wrapped(double scale, const RectSet& bounds,
                                                   bool relative_to_bounds = false) const;
  /**
   * Scale the set by the given factor while keeping it inside the given bounds.
   * If any dimension exceeds the bounds after scaling, it is wrapped around to the other side,
   * as another set.
   * The scaling is performed with respect to the center of the set.
   * The scaling factor can be computed relative to either
   * - the current size of the set;
   * - the size of the bounding rectangular set.
   * @param scale scaling factor per dimension
   * @param bounds bounding rectangular set
   * @param relative_to_bounds if true, the scaling factor is computed relative to the size of the bounding
   * rectangular set; if false, the scaling factor is computed relative to the current size of the rectangular
   * @return new scaled rectangular set
   */
  [[nodiscard]] std::unique_ptr<Set> scale_wrapped(ConstVectorRef scale, const RectSet& bounds,
                                                   bool relative_to_bounds = false) const;

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
   * Convert the set to a rectangular set.
   * Not all sets can be converted to rectangular sets.
   * If the set cannot be converted, an exception is thrown.
   * @return unique pointer to the rectangular set
   */
  [[nodiscard]] virtual std::unique_ptr<Set> to_rect_set() const;

  /** @getter{lower bound, rectangular set} */
  [[nodiscard]] virtual Vector general_lower_bound() const;
  /** @getter{upper bound, rectangular set} */
  [[nodiscard]] virtual Vector general_upper_bound() const;

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

  virtual bool operator==(const Set& other) const;
  bool operator!=(const Set& other) const = default;

  /**
   * Get string representation of the set.
   * @return string representation
   */
  /** @to_string */
  [[nodiscard]] virtual std::string to_string() const;

 protected:
  /**
   * Scale the rectangular set by the given factor while keeping it inside the given bounds.
   * If any dimension exceeds the bounds after scaling, it is wrapped around to the other side,
   * as another rectangular set.
   * The scaling is performed with respect to the center of the rectangular set.
   * The scaling factor can be computed relative to either
   * - the current size of the rectangular set;
   * - the size of the bounding rectangular set.
   * @param scale scaling factor
   * @param bounds bounding rectangular set
   * @param relative_to_bounds if true, the scaling factor is computed relative to the size of the bounding
   * rectangular set; if false, the scaling factor is computed relative to the current size of the rectangular
   * @return new scaled rectangular set
   */
  [[nodiscard]] virtual std::unique_ptr<Set> scale_wrapped_impl(ConstVectorRef scale, const RectSet& bounds,
                                                                bool relative_to_bounds) const;
};

std::ostream& operator<<(std::ostream& os, const Set& set);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Set)

#endif
