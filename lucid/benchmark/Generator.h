/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Generator class.
 */
#pragma once

#include "lucid/lib/eigen.h"
#include "lucid/math/Set.h"

namespace lucid::benchmark {

/**
 * Generate a problem for the solver.
 * A generator defines a vector space @X and a function @f$ f: \mathcal{X} \to \mathcal{X} @f$.
 */
class Generator {
 public:
  virtual ~Generator() = default;
  /** @getter{dimension, vector space @X} */
  [[nodiscard]] virtual Dimension dimension() const = 0;
  [[nodiscard]] virtual Matrix operator()(ConstMatrixRef x) const = 0;
  [[nodiscard]] Matrix transition(ConstMatrixRef x) const { return (*this)(x); }
  [[nodiscard]] virtual int num_steps() const = 0;
  [[nodiscard]] virtual double desired_confidence() = 0;
  /** @getter{initial set, vector space @X} */
  [[nodiscard]] virtual const Set& initial_set() const = 0;
  /** @getter{unsafe set, vector space @X} */
  [[nodiscard]] virtual const Set& unsafe_set() const = 0;
  /** @getter{all sets we are interested in considering, vector space @X} */
  [[nodiscard]] virtual const Set& set() const = 0;
  /**
   * Sample `num_samples` transitions from the generator.
   * @param num_samples number of samples to generate
   * @param[out] inputs `n` x `num_samples` matrix of samples, where `n` is the dimension of the vector space @X.
   * Each column is a sample from the set.
   * @param[out] outputs `n` x `num_samples` matrix of outputs, where `n` is the dimension of the vector space @X
   * Each column is obtained by `outputs.col(i) = apply(inputs.col(i))`
   */
  void sample_transition(int num_samples, Matrix& inputs, Matrix& outputs) const;
  /**
   * Sample a transition from the generator.
   * @param[out] input `n` vector, a sample from the whole set.
   * @param[out] output `n` vector obtained by `output = apply(input)`
   */
  void sample_transition(Vector& input, Vector& output) const;
  /**
   * Sample `num_samples` elements from the generator set.
   * @param num_samples number of samples to generate
   * @return `n` x `num_samples` matrix of samples, where `n` is the dimension of the vector space @X
   */
  Matrix sample_element(int num_samples) const;
  /**
   * Sample an element from the generator set.
   * @return `n` vector, a sample from the whole set.
   */
  Vector sample_element() const;

  /** Plot the generator information using matplotlib. */
  void plot() const;
};

}  // namespace lucid::benchmark
