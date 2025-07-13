import type { Configuration } from "@/utils/schema";

export const emptyFigure = { data: [], layout: {} };

export const defaultValues: Configuration = {
  verbose: 3,
  seed: -1,
  x_samples: [],
  xp_samples: [],
  system_dynamics: [],
  X_bounds: [
    {
      RectSet: [[0, 1]],
    },
  ],
  X_init: [
    {
      RectSet: [[0, 1]],
    },
  ],
  X_unsafe: [
    {
      RectSet: [[0, 1]],
    },
  ],
  gamma: 1.0,
  c_coefficient: 1.0,
  lambda: 1.0,
  num_samples: 1000,
  time_horizon: 5,
  sigma_f: 15.0,
  sigma_l: 1.0,
  num_frequencies: 4,
  oversample_factor: 2.0,
  num_oversample: -1,
  noise_scale: 0.01,
  plot: false,
  verify: true,
  problem_log_file: "problem.lp",
  iis_log_file: "iis.ilp",
  estimator: "KernelRidgeRegressor",
  kernel: "GaussianKernel",
  feature_map: "LinearTruncatedFourierFeatureMap",
  optimiser: "GurobiOptimiser",
};
