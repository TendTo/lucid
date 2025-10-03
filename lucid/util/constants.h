/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Constants values.
 * Usually dependent on the build configuration.
 */
#pragma once

/**
 * @namespace lucid::constants
 * Constants values used in the lucid library.
 * Usually dependent on the build configuration.
 */
namespace lucid::constants {

// NOLINTNEXTLINE: false positive
/** Whether Lucid is built with the [Gurobi](https://www.gurobi.com/) library. */
constexpr bool GUROBI_BUILD =
#ifdef LUCID_GUROBI_BUILD
    true;
#else
    false;
#endif

// NOLINTNEXTLINE: false positive
/** Whether Lucid is built with the [Alglib](https://www.alglib.net/) library. */
constexpr bool ALGLIB_BUILD =
#ifdef LUCID_ALGLIB_BUILD
    true;
#else
    false;
#endif

// NOLINTNEXTLINE: false positive
/** Whether Lucid is built with the [Highs](https://highs.dev/) library. */
constexpr bool HIGHS_BUILD =
#ifdef LUCID_HIGHS_BUILD
    true;
#else
    false;
#endif

// NOLINTNEXTLINE: false positive
/** Whether Lucid is built with the [Soplex](https://soplex.zib.de/) library. */
constexpr bool SOPLEX_BUILD =
#ifdef LUCID_SOPLEX_BUILD
    true;
#else
    false;
#endif

// NOLINTNEXTLINE: false positive
/** Whether Lucid is built with the [Matplotlib](https://matplotlib.org/) library. */
constexpr bool MATPLOTLIB_BUILD =
#ifdef LUCID_MATPLOTLIB_BUILD
    true;
#else
    false;
#endif

// NOLINTNEXTLINE: false positive
/** Whether Lucid is built with [CUDA](https://developer.nvidia.com/cuda-toolkit) support. */
constexpr bool CUDA_BUILD =
#ifdef LUCID_CUDA_BUILD
    true;
#else
    false;
#endif

// NOLINTNEXTLINE: false positive
/** Whether Lucid is built with [OpenMP](https://www.openmp.org/) support. */
constexpr bool OMP_BUILD =
#ifdef LUCID_OMP_BUILD
    true;
#else
    false;
#endif

}  // namespace lucid::constants
