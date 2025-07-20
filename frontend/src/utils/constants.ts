import type { Configuration } from "@/utils/schema";

export const emptyFigure = { data: [], layout: {} };

export const defaultValues: Configuration = {
  verbose: 3,
  seed: -1,
  dimension: 1,
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
  plot: true,
  verify: true,
  problem_log_file: "",
  iis_log_file: "",
  estimator: "KernelRidgeRegressor",
  kernel: "GaussianKernel",
  feature_map: "LinearTruncatedFourierFeatureMap",
  optimiser: "GurobiOptimiser",
};

const Linear: Partial<Configuration> = {
  verbose: 3,
  seed: 42,
  dimension: 1,
  system_dynamics: ["x1 / 2"],
  x_samples: [],
  xp_samples: [],
  X_bounds: [{ RectSet: [[-1, 1]] }],
  X_init: [{ RectSet: [[-0.5, 0.5]] }],
  X_unsafe: [
    {
      RectSet: [[-1, -0.9]],
    },
    {
      RectSet: [[0.9, 1]],
    },
  ],
  gamma: 1,
  c_coefficient: 1,
  lambda: 0.001,
  num_samples: 1000,
  time_horizon: 5,
  sigma_f: 15.0,
  sigma_l: 1.75555556,
  num_frequencies: 8,
  oversample_factor: 128.0,
  num_oversample: -1,
  noise_scale: 0.01,
  plot: true,
  verify: true,
  estimator: "KernelRidgeRegressor",
  feature_map: "LinearTruncatedFourierFeatureMap",
  optimiser: "GurobiOptimiser",
};

const Barrier3: Partial<Configuration> = {
  verbose: 3,
  seed: 42,
  dimension: 2,
  system_dynamics: ["x2", "-x1 - x2 + 1 / 3 * x1 ** 3"],
  x_samples: [],
  xp_samples: [],
  X_bounds: [
    {
      RectSet: [
        [-3, 2.5],
        [-2, 1],
      ],
    },
  ],
  X_init: [
    {
      RectSet: [
        [1, 2],
        [-0.5, 0.5],
      ],
    },
  ],
  X_unsafe: [
    {
      RectSet: [
        [-2.9, -2.8],
        [0.1, 0.5],
      ],
    },
  ],
  gamma: 2,
  c_coefficient: 1,
  lambda: 0.000001,
  num_samples: 1000,
  time_horizon: 5,
  sigma_f: 15,
  sigma_l: 1.75555556,
  num_frequencies: 4,
  oversample_factor: 32,
  num_oversample: -1,
  noise_scale: 0.01,
  plot: true,
  verify: true,
  estimator: "KernelRidgeRegressor",
  feature_map: "LinearTruncatedFourierFeatureMap",
  optimiser: "GurobiOptimiser",
  kernel: "GaussianKernel",
};

export const examples: Record<
  string,
  { config: Partial<Configuration>; name: string }
> = {
  linear: { config: Linear, name: "Linear System" },
  barrier3: { config: Barrier3, name: "Barrier 3" },
};

export const availableSets = {
  RectSet: "Rect Set",
  SphereSet: "Sphere Set",
};
export const defaultSet: Record<keyof typeof availableSets, object> = {
  RectSet: [[0, 1]],
  SphereSet: { center: [0], radius: 1 },
} as const;
