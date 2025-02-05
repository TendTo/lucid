/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Set class.
 */
#pragma once

#include <iosfwd>

#include "lucid/lib/eigen.h"

namespace lucid {

/**
 * Generic set over an @f$ n @f$ dimensional vector space @f$ \mathcal{X} \in \mathbb{R}^n @f$.
 * It can be used to test if a vector is in the set and to sample elements from the set.
 */
class Set {
 public:
  virtual ~Set() = default;
  /** @getter{dimension, vector space @X} */
  [[nodiscard]] virtual Dimension dimension() const = 0;
  /**
   * Extract `num_samples` elements of the set @X using some kind of random distribution.
   * @param num_samples number of samples to generate
   * @return `n` x `num_samples` matrix of samples, where `n` is the dimension of the vector space.
   * In other words, the samples are stored as column vectors.
   */
  [[nodiscard]] virtual Matrix sample_element(int num_samples) const = 0;
  /**
   * Extract an element of the set @X using some kind of random distribution.
   * @return element of the set
   */
  [[nodiscard]] Vector sample_element() const;
  /**
   * Check if a vector is in the set @X.
   * @param x vector to test
   * @return true if x is in the set
   * @return false if x is not in the set
   * @throw LucidInvalidArgument if x does not belong to the vector space of the set
   */
  [[nodiscard]] bool contains(ConstMatrixRef x) const { return (*this)(x); }

  /**
   * Check if a vector is in the set @X.
   * @param x vector to test
   * @return true if x is in the set
   * @return false if x is not in the set
   * @throw LucidInvalidArgument if x does not belong to the vector space of the set
   */
  [[nodiscard]] virtual bool operator()(ConstMatrixRef x) const = 0;

  /** Plot the set information using matplotlib. */
  virtual void plot(const std::string& color) const = 0;
};

std::ostream& operator<<(std::ostream& os, const Set& set);

}  // namespace lucid
