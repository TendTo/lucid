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
/** Whether Lucid is built with the [pso-cpp](https://github.com/Rookfighter/pso-cpp/tree/master) library. */
constexpr bool PSOCPP_BUILD =
#ifdef LUCID_PSOCPP_BUILD
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

// NOLINTNEXTLINE: false positive
/** Whether Lucid is built in debug mode. */
constexpr bool DEBUG_BUILD =
#ifndef NDEBUG
    true;
#else
    false;
#endif

// NOLINTNEXTLINE: false positive
/** Whether Lucid is built in relase mode. */
constexpr bool RELEASE_BUILD =
#ifdef NDEBUG
    true;
#else
    false;
#endif

// NOLINTNEXTLINE: false positive
/**
 * Whether runtime checks are enabled.
 * Runtime checks are responsible for throwing a LucidInvalidArgumentException when function preconditions are violated.
 */
constexpr bool RUNTIME_CHECKS_ENABLED =
#ifndef NCHECK
    true;
#else
    false;
#endif

// NOLINTNEXTLINE: false positive
/**
 * Whether runtime assertions are enabled.
 * Assertions are possibly expensive internal consistency checks used to verify the correctness of the code.
 */
constexpr bool ASSERT_CHECKS_ENABLED =
#ifndef NDEBUG
    true;
#else
    false;
#endif

// NOLINTNEXTLINE: false positive
/** Whether logging is enabled. */
constexpr bool LOG_ENABLED =
#ifndef NLOG
    true;
#else
    false;
#endif

}  // namespace lucid::constants
