import { z } from "zod";

const set = z
  .array(
    z.object({
      RectSet: z
        .array(
          z.tuple([z.number(), z.number()]).refine(([min, max]) => min < max, {
            message: "Inverted bounds.",
          })
        )
        .nonempty()
        .describe(
          "Bounds for the state space, each tuple is a [lower, upper] bound."
        ),
    })
  )
  .nonempty()
  .refine(
    (rectSet) =>
      rectSet.every((r) => r.RectSet.length === rectSet[0].RectSet.length),
    {
      message: "All sets must have the same dimension.",
    }
  );
const matrix = z
  .array(z.array(z.any()))
  .refine(
    (samples) => {
      let expectedColumns: number | null = null;
      for (const sample of samples) {
        expectedColumns ??= sample.length;
        if (sample.length !== expectedColumns) {
          return false;
        }
      }
      return true;
    },
    {
      message: "Column size mismatch",
    }
  )
  .refine(
    (samples) => {
      for (const sample of samples) {
        if (
          sample.filter((x) => isNaN(x) || typeof x !== "number").length > 0
        ) {
          return false;
        }
      }
      return true;
    },
    {
      message: "Failed to parse samples",
    }
  );
export const configurationSchema = z
  .object({
    verbose: z
      .number()
      .int()
      .gte(-1)
      .lte(5)
      .describe("Verbosity level of the output.")
      .default(3),
    seed: z
      .number()
      .int()
      .describe("Seed for the random number generator.")
      .gte(-1)
      .default(-1),
    dimension: z
      .number()
      .int()
      .gte(1)
      .lte(9)
      .describe("Dimension of the state space.")
      .default(1),
    x_samples: matrix.describe(
      "Samples of the state space, each sample is an array of numbers."
    ),
    xp_samples: matrix.describe(
      "Samples of the state space at the next time step, each sample is an array of numbers."
    ),
    system_dynamics: z
      .array(z.string().nonempty())
      .describe("Dynamics of the system, describing how it evolves over time."),
    X_bounds: set,
    X_init: set,
    X_unsafe: set,
    gamma: z.number(),
    c_coefficient: z.number().gte(0),
    lambda: z.number(),
    num_samples: z
      .number()
      .int()
      .gte(1)
      .describe("Number of samples to use to fit the estimator.")
      .default(1000),
    time_horizon: z
      .number()
      .int()
      .gte(1)
      .describe("Time horizon to consider in the specification.")
      .default(5),
    sigma_f: z.number(),
    sigma_l: z
      .union([z.number(), z.array(z.number())])
      .describe(
        "Length scale for the feature map, can be a single value or a list."
      ),
    num_frequencies: z
      .number()
      .int()
      .gte(1)
      .describe("Number of frequencies to use in the feature map.")
      .default(4),
    oversample_factor: z
      .number()
      .describe("Factor by which to oversample the feature map."),
    num_oversample: z
      .number()
      .int()
      .describe("Number of oversamples to use, -1 for no oversampling."),
    noise_scale: z
      .number()
      .gte(0)
      .describe("Scale of the noise to add to the system dynamics."),
    plot: z.boolean().describe("Whether to plot the results."),
    verify: z.boolean().describe("Whether to verify the results."),
    problem_log_file: z
      .string()
      .describe("File to log the problem formulation.")
      .default("problem.lp"),
    iis_log_file: z
      .string()
      .describe("File to log the irreducible infeasible set (IIS).")
      .default("iis.ilp"),
    estimator: z
      .literal("KernelRidgeRegressor")
      .describe("Type of estimator to use.")
      .default("KernelRidgeRegressor"),
    kernel: z
      .literal("GaussianKernel")
      .describe("Type of kernel to use for the estimator.")
      .default("GaussianKernel"),
    feature_map: z
      .enum([
        "LinearTruncatedFourierFeatureMap",
        "ConstantTruncatedFourierFeatureMap",
        "LogTruncatedFourierFeatureMap",
      ])
      .describe("Type of feature map to use for the estimator.")
      .default("LinearTruncatedFourierFeatureMap"),
    optimiser: z
      .enum(["GurobiOptimiser", "AlglibOptimiser", "HighsOptimiser"])
      .describe("Type of optimiser to use for the estimator.")
      .default("GurobiOptimiser"),
  })
  .strict()
  // [Enforced by dimension]
  // .superRefine((data, ctx) => {
  //   const inputDimensions = data.X_bounds.length
  //     ? data.X_bounds[0].RectSet.length
  //     : 0;
  //   const valid =
  //     data.X_bounds.every((r) => r.RectSet.length === inputDimensions) &&
  //     data.X_init.every((r) => r.RectSet.length === inputDimensions) &&
  //     data.X_unsafe.every((r) => r.RectSet.length === inputDimensions);
  //   if (valid) return;
  //   for (const key of ["X_bounds", "X_init", "X_unsafe"] as const) {
  //     ctx.addIssue({
  //       path: [key],
  //       code: "custom",
  //       message: "X_bounds, X_init, and X_unsafe must have the same dimension.",
  //     });
  //   }
  // })
  .superRefine((data, ctx) => {
    const inputDimensions = data.X_bounds.length
      ? data.X_bounds[0].RectSet.length
      : 0;
    if (data.system_dynamics.length === 0) return;
    if (inputDimensions === 0) return;
    const xs = new Set();
    for (const f of data.system_dynamics ?? []) {
      const match = f.matchAll(/x(\d+)/g);
      for (const m of match ?? []) {
        const x = parseInt(m[1], 10);
        xs.add(x);
      }
    }
    const expectedValues = Array.from(Array(inputDimensions).keys());
    if (
      xs.size !== inputDimensions ||
      expectedValues.some((v) => !xs.has(v + 1))
    ) {
      ctx.addIssue({
        path: ["system_dynamics"],
        code: "custom",
        message: `The model must reference all and only input components [x1, ..., x${inputDimensions}].`,
      });
    }
  })
  .superRefine((data, ctx) => {
    if (data.system_dynamics.length === 0) return;
    if (data.x_samples.length == 0 && data.xp_samples.length == 0) return;
    for (const cause of [
      "system_dynamics",
      "x_samples",
      "xp_samples",
    ] as const) {
      ctx.addIssue({
        path: [cause],
        code: "custom",
        message: `You can either define the model or provide samples, not both.`,
      });
    }
  })
  .superRefine((data, ctx) => {
    if (data.system_dynamics.length !== 0) return;
    if (data.x_samples.length > 0 && data.xp_samples.length > 0) return;
    for (const cause of [
      "system_dynamics",
      "x_samples",
      "xp_samples",
    ] as const) {
      if (data[cause].length > 0) continue;
      ctx.addIssue({
        path: [cause],
        code: "custom",
        message: `You must provide either model or samples.`,
      });
    }
  })
  .superRefine((data, ctx) => {
    if (data.x_samples.length === data.xp_samples.length) return;
    for (const cause of ["x_samples", "xp_samples"] as const) {
      ctx.addIssue({
        path: [cause],
        code: "custom",
        message: `Number of samples mismatch`,
      });
    }
  })
  .superRefine((data, ctx) => {
    const sampleDimension = data.x_samples.at(0)?.length ?? 0;
    if (sampleDimension === 0 || sampleDimension === data.dimension) return;
    for (const cause of ["x_samples", "dimension"] as const) {
      ctx.addIssue({
        path: [cause],
        code: "custom",
        message: `Samples dimension mismatch (${sampleDimension} != ${data.dimension}).`,
      });
    }
  })
  .superRefine((data, ctx) => {
    if (
      typeof data.sigma_l === "number" ||
      data.sigma_l.length === 1 ||
      data.sigma_l.length === data.dimension
    )
      return;
    ctx.addIssue({
      path: ["sigma_l"],
      code: "custom",
      message: `Must be a number or an array of length ${data.dimension}.`,
    });
  })
  .describe(
    "Representation of the command line arguments for pylucid expressed in a configuration file"
  );

export type Configuration = z.infer<typeof configurationSchema>;
