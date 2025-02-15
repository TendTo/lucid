/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/benchmark/Scenario.h"

namespace lucid::benchmark {

void Scenario::sample_transition(const int num_samples, Matrix& inputs, Matrix& outputs) const {
  inputs = set().sample_element(num_samples);
  outputs = Matrix::Zero(num_samples, dimension());
  for (int i = 0; i < num_samples; i++) outputs.row(i) = transition(inputs.row(i)).transpose();
}

void Scenario::sample_transition(Vector& input, Vector& output) const {
  Matrix inputs, outputs;
  sample_transition(1, inputs, outputs);
  input = inputs.col(0);
  output = outputs.col(0);
}
Matrix Scenario::sample_element(const int num_samples) const { return set().sample_element(num_samples); }
Vector Scenario::sample_element() const { return set().sample_element(); }

void Scenario::plot() const {
  set().plot("black");
  initial_set().plot("blue");
  unsafe_set().plot("red");
}

}  // namespace lucid::benchmark
