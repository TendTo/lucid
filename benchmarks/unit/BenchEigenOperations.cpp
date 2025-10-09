/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Benchmark for pseudo-inverse computation methods.
 */
#include <benchmark/benchmark.h>

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
BENCHMARK(UseComparisonAllFalse)->LUCID_RUNS;
BENCHMARK(UseMinFalse)->LUCID_RUNS;
BENCHMARK(UseOMPNestedLoopFalse)->LUCID_RUNS;
BENCHMARK(UseOMPSingleLoopFalse)->LUCID_RUNS;
