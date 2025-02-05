/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/benchmark/Generator.h"

namespace lucid::benchmark {

void Generator::sample_transition(const int num_samples, Matrix& inputs, Matrix& outputs) const {
  inputs = set().sample_element(num_samples);
  outputs = Matrix::Zero(dimension(), num_samples);
  for (int i = 0; i < num_samples; i++) outputs.col(i) = transition(inputs.col(i));
}

void Generator::sample_transition(Vector& input, Vector& output) const {
  Matrix inputs, outputs;
  sample_transition(1, inputs, outputs);
  input = inputs.col(0);
  output = outputs.col(0);
}
Matrix Generator::sample_element(const int num_samples) const { return set().sample_element(num_samples); }
Vector Generator::sample_element() const { return set().sample_element(); }

void Generator::plot() const {
  set().plot("black");
  initial_set().plot("blue");
  unsafe_set().plot("red");
}

}  // namespace lucid::benchmark
