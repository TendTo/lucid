/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Scenario class.
 */
#pragma once

#include "lucid/lib/eigen.h"
#include "lucid/math/Set.h"

namespace lucid::benchmark {

/**
 * Generate a problem for the solver.
 * A scenario defines a vector space @X and a transition function @f$ f: \mathcal{X} \to \mathcal{X} @f$.
 */
class Scenario {
 public:
  virtual ~Scenario() = default;
  /** @getter{dimension, vector space @X} */
  [[nodiscard]] virtual Dimension dimension() const = 0;
  /**
   * Apply the transition function to a vector of the vector space @X.
   * @param x @nvector from @X
   * @return @nvector @fx
   */
  [[nodiscard]] virtual Matrix operator()(ConstMatrixRef x) const = 0;
  /**
   * Apply the transition function to a vector of the vector space @X.
   * @param x @nvector from @X
   * @return @nvector @fx
   */
  [[nodiscard]] Matrix transition(ConstMatrixRef x) const { return (*this)(x); }
  /** @getter{number of steps accounted for, scenario output state} */
  [[nodiscard]] virtual int num_steps() const = 0;
  /** @getter{desired confidence level, scenario} */
  [[nodiscard]] virtual double desired_confidence() = 0;
  /** @getter{initial set, vector space @X} */
  [[nodiscard]] virtual const Set& initial_set() const = 0;
  /** @getter{unsafe set, vector space @X} */
  [[nodiscard]] virtual const Set& unsafe_set() const = 0;
  /** @getter{all sets we are interested in considering, vector space @X} */
  [[nodiscard]] virtual const Set& set() const = 0;
  /**
   * Sample `num_samples` transitions from the scenario.
   * @param num_samples number of samples to generate
   * @param[out] inputs `n` x `num_samples` matrix of samples, where `n` is the dimension of the vector space @X.
   * Each column is a sample from the set.
   * @param[out] outputs `n` x `num_samples` matrix of outputs, where `n` is the dimension of the vector space @X
   * Each column is obtained by `outputs.col(i) = apply(inputs.col(i))`
   */
  void sample_transition(int num_samples, Matrix& inputs, Matrix& outputs) const;
  /**
   * Sample a transition from the scenario.
   * @param[out] input @nvector, a sample from the whole set.
   * @param[out] output @nvector from `apply(input)`
   */
  void sample_transition(Vector& input, Vector& output) const;
  /**
   * Sample `num_samples` elements from the scenario set.
   * @param num_samples number of samples to generate
   * @return `n` x `num_samples` matrix of samples, where `n` is the dimension of the vector space @X
   */
  [[nodiscard]] Matrix sample_element(Index num_samples) const;
  /**
   * Sample an element from the scenario set.
   * @return @nvector, a sample from the whole set.
   */
  [[nodiscard]] Vector sample_element() const;

  /** Plot the scenario information using matplotlib. */
  void plot() const;
};

}  // namespace lucid::benchmark
