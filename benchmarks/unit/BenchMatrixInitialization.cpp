/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Benchmark for matrix initialization alternatives.
 */
#include <benchmark/benchmark.h>

#include "lucid/lib/eigen.h"

using lucid::Index;
using lucid::Matrix;
using lucid::Vector;

constexpr unsigned int num_frequencies = 10;
constexpr double sigma_l = 1.4;

void UseNullaryExpression(benchmark::State& state) {
  for (auto _ : state) {
    Matrix prob_per_dim{state.range(0), state.range(0)};
    for (Index i = 0; i < state.range(0); i++) {
      constexpr double offset = 3 * sigma_l / (2 * num_frequencies - 1);
      Vector intervals{Vector::NullaryExpr(
          num_frequencies + 1, [](const Index idx) { return static_cast<double>(2 * idx - 1) * offset; })};
      // Reset the first interval to 0 instead of -offset
      intervals(0) = 0;
      prob_per_dim.row(i) = lucid::normal_cdf(intervals.tail(num_frequencies), 0, sigma_l) -
                            lucid::normal_cdf(intervals.head(num_frequencies), 0, sigma_l);
      prob_per_dim.row(i) *= 2;
      benchmark::DoNotOptimize(prob_per_dim.sum());
    }
  }
  state.SetComplexityN(state.range(0));
}

void UseLoopExpression(benchmark::State& state) {
  for (auto _ : state) {
    Matrix prob_per_dim{state.range(0), state.range(0)};
    for (Index i = 0; i < state.range(0); i++) {
      constexpr double offset = 3 * sigma_l / (2 * num_frequencies - 1);
      Vector intervals{num_frequencies + 1};
      intervals(0) = 0;
      intervals(1) = offset;
      for (Index idx = 2; idx < num_frequencies + 1; idx++) intervals(idx) = intervals(idx - 1) + 2 * offset;
      // Reset the first interval to 0 instead of -offset
      prob_per_dim.row(i) = lucid::normal_cdf(intervals.tail(num_frequencies), 0, sigma_l) -
                            lucid::normal_cdf(intervals.head(num_frequencies), 0, sigma_l);
      prob_per_dim.row(i) *= 2;
      benchmark::DoNotOptimize(prob_per_dim.sum());
    }
  }
  state.SetComplexityN(state.range(0));
}

void UseOMPLoopExpression(benchmark::State& state) {
  for (auto _ : state) {
    Matrix prob_per_dim{state.range(0), state.range(0)};
#pragma omp parallel for
    for (Index i = 0; i < state.range(0); i++) {
      constexpr double offset = 3 * sigma_l / (2 * num_frequencies - 1);
      Vector intervals{num_frequencies + 1};
      intervals(0) = 0;
      intervals(1) = offset;
      for (Index idx = 2; idx < num_frequencies + 1; idx++) intervals(idx) = intervals(idx - 1) + 2 * offset;
      // Reset the first interval to 0 instead of -offset
      prob_per_dim.row(i) = lucid::normal_cdf(intervals.tail(num_frequencies), 0, sigma_l) -
                            lucid::normal_cdf(intervals.head(num_frequencies), 0, sigma_l);
      prob_per_dim.row(i) *= 2;
      benchmark::DoNotOptimize(prob_per_dim.sum());
    }
  }
  state.SetComplexityN(state.range(0));
}

#define LUCID_RUNS Range(1 << 4, 1 << 10)->Complexity(benchmark::oNSquared)

BENCHMARK(UseNullaryExpression)->LUCID_RUNS;
BENCHMARK(UseLoopExpression)->LUCID_RUNS;
BENCHMARK(UseOMPLoopExpression)->LUCID_RUNS;
