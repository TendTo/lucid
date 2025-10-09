/**
 * @author Ernesto Casablanca
 * @author Oliver Schön
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Benchmark for pseudo-inverse computation methods.
 */
#include <benchmark/benchmark.h>

#include <execution>

#include "lucid/lib/eigen.h"

using lucid::Index;
using lucid::Matrix;
using lucid::Vector;

constexpr unsigned int num_frequencies = 10;
constexpr double sigma_l = 1.4;

void UseComparisonAllTrue(benchmark::State& state) {
  const Matrix mat{Matrix::Random(state.range(0), state.range(0))};
  for (auto _ : state) {
    bool res = (mat.array() > -2).all();
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

void UseMinTrue(benchmark::State& state) {
  const Matrix mat{Matrix::Random(state.range(0), state.range(0))};
  for (auto _ : state) {
    bool res = mat.minCoeff() > -2;
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

void UseOMPNestedLoopTrue(benchmark::State& state) {
  const Matrix mat{Matrix::Random(state.range(0), state.range(0))};
  for (auto _ : state) {
    bool res = true;
#pragma omp parallel for collapse(2) shared(res)
    for (Index i = 0; i < mat.rows(); i++) {
      for (Index j = 0; j < mat.cols(); j++) {
        if (mat(i, j) <= -2) res = false;
      }
    }
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

void UseOMPSingleLoopTrue(benchmark::State& state) {
  const Matrix mat{Matrix::Random(state.range(0), state.range(0))};
  for (auto _ : state) {
    bool res = true;
#pragma omp parallel for shared(res)
    for (Index i = 0; i < mat.size(); i++) {
      if (mat.data()[i] <= -2) res = false;
    }
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

template <class Policy>
  requires std::is_execution_policy<Policy>::value
void UseTransformReduceLoopTrue(benchmark::State& state) {
  constexpr Policy policy{};
  const Matrix mat{Matrix::Random(state.range(0), state.range(0))};
  std::span<const double> data_span{mat.data(), static_cast<std::size_t>(mat.size())};
  for (auto _ : state) {
    bool res = std::transform_reduce(policy, data_span.begin(), data_span.end(), true, std::logical_and<>{},
                                     [](const double val) { return val > -2; });
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

void UseComparisonAllFalse(benchmark::State& state) {
  const Matrix mat{Matrix::Random(state.range(0), state.range(0))};
  for (auto _ : state) {
    bool res = (mat.array() > 0).all();
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

void UseMinFalse(benchmark::State& state) {
  const Matrix mat{Matrix::Random(state.range(0), state.range(0))};
  for (auto _ : state) {
    bool res = mat.minCoeff() > 0;
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

template <class Policy>
  requires std::is_execution_policy<Policy>::value
void UseTransformReduceLoopFalse(benchmark::State& state) {
  constexpr Policy policy{};
  const Matrix mat{Matrix::Random(state.range(0), state.range(0))};
  std::span<const double> data_span{mat.data(), static_cast<std::size_t>(mat.size())};
  for (auto _ : state) {
    bool res = std::transform_reduce(policy, data_span.begin(), data_span.end(), true, std::logical_and<>{},
                                     [](const double val) { return val > 0; });
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

void UseOMPNestedLoopFalse(benchmark::State& state) {
  const Matrix mat{Matrix::Random(state.range(0), state.range(0))};
  for (auto _ : state) {
    bool res = true;
#pragma omp parallel for collapse(2) shared(res)
    for (Index i = 0; i < mat.rows(); i++) {
      for (Index j = 0; j < mat.cols(); j++) {
        if (mat(i, j) <= 0) res = false;
      }
    }
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

void UseOMPSingleLoopFalse(benchmark::State& state) {
  const Matrix mat{Matrix::Random(state.range(0), state.range(0))};
  for (auto _ : state) {
    bool res = true;
#pragma omp parallel for shared(res)
    for (Index i = 0; i < mat.size(); i++) {
      if (mat.data()[i] <= 0) res = false;
    }
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(state.range(0));
}

#define LUCID_RUNS Range(1 << 4, 1 << 10)->Complexity(benchmark::oNSquared)

BENCHMARK(UseComparisonAllTrue)->LUCID_RUNS;
BENCHMARK(UseMinTrue)->LUCID_RUNS;
BENCHMARK(UseOMPNestedLoopTrue)->LUCID_RUNS;
BENCHMARK(UseOMPSingleLoopTrue)->LUCID_RUNS;
BENCHMARK(UseTransformReduceLoopTrue<std::execution::sequenced_policy>)->LUCID_RUNS;
BENCHMARK(UseTransformReduceLoopTrue<std::execution::unsequenced_policy>)->LUCID_RUNS;
BENCHMARK(UseTransformReduceLoopTrue<std::execution::parallel_policy>)->LUCID_RUNS;
BENCHMARK(UseTransformReduceLoopTrue<std::execution::parallel_unsequenced_policy>)->LUCID_RUNS;
BENCHMARK(UseComparisonAllFalse)->LUCID_RUNS;
BENCHMARK(UseMinFalse)->LUCID_RUNS;
BENCHMARK(UseOMPNestedLoopFalse)->LUCID_RUNS;
BENCHMARK(UseOMPSingleLoopFalse)->LUCID_RUNS;
BENCHMARK(UseTransformReduceLoopFalse<std::execution::sequenced_policy>)->LUCID_RUNS;
BENCHMARK(UseTransformReduceLoopFalse<std::execution::unsequenced_policy>)->LUCID_RUNS;
BENCHMARK(UseTransformReduceLoopFalse<std::execution::parallel_policy>)->LUCID_RUNS;
BENCHMARK(UseTransformReduceLoopFalse<std::execution::parallel_unsequenced_policy>)->LUCID_RUNS;
