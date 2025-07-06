import { FaFileImport } from "react-icons/fa6";
import type { FormStepName, FormSteps } from "@app/types/types";
import type { FieldValues, UseFormReset } from "react-hook-form";

export type HeaderProps = {
  errors: object;
  steps: FormSteps;
  setCurrentStep: (step: FormStepName) => void;
  setIsImportModalOpen: (isOpen: boolean) => void; // Optional for import modal
};

const Linear = {
  verbose: 3,
  seed: 42,
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
  sigma_f: 15,
  sigma_l: 1.75555556,
  num_frequencies: 4,
  oversample_factor: 32,
  num_oversample: -1,
  noise_scale: 0.01,
  plot: true,
  verify: true,
  problem_log_file: "problem.lp",
  iis_log_file: "iis.ilp",
  estimator: "KernelRidgeRegressor",
  feature_map: "LinearTruncatedFourierFeatureMap",
  optimiser: "GurobiOptimiser",
} as const;

const Barrier3 = {
  verbose: 3,
  seed: 42,
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
  problem_log_file: "problem.lp",
  iis_log_file: "iis.ilp",
  estimator: "KernelRidgeRegressor",
  feature_map: "LinearTruncatedFourierFeatureMap",
  optimiser: "GurobiOptimiser",
  kernel: "GaussianKernel",
} as const;

type ExamplesProps = {
  reset: UseFormReset<FieldValues>;
};

export default function Examples({ reset }: ExamplesProps) {
  return (
    <>
      <button
        type="button"
        onClick={() => reset(Linear)}
        className="btn btn-secondary flex items-center justify-center"
      >
        <FaFileImport className="inline-block mr-1" />
        Linear
      </button>
      <button
        type="button"
        onClick={() => reset(Barrier3)}
        className="btn btn-secondary flex items-center justify-center"
      >
        <FaFileImport className="inline-block mr-1" />
        Barrier3
      </button>
    </>
  );
}
