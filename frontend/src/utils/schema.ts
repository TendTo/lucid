import { z } from "zod";

const set = z
  .array(
    z.object({
      RectSet: z
        .array(
          z
            .tuple([z.number(), z.number()])
            .refinement(([min, max]) => min < max, {
              message: "Inverted bounds.",
              code: "custom",
            })
        )
        .nonempty()
        .describe(
          "Bounds for the state space, each tuple is a [lower, upper] bound."
        ),
    })
  )
  .nonempty()
  .refinement(
    (rectSet) =>
      rectSet.every((r) => r.RectSet.length === rectSet[0].RectSet.length),
    {
      message: "All sets must have the same dimension.",
      code: "custom",
    }
  );
const matrix = z
  .array(z.array(z.any()))
  .refinement(
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
      code: "custom",
    }
  )
  .refinement(
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
      code: "custom",
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
    gamma: z.number().optional(),
    c_coefficient: z.coerce.number().gte(0).optional(),
    lambda: z.number().optional(),
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
    sigma_f: z.number().optional(),
    sigma_l: z
      .any()
      .superRefine((x, ctx) => {
        const schemas = [z.number(), z.array(z.number())];
        const errors = schemas.reduce<z.ZodError[]>(
          (errors, schema) =>
            ((result) => (result.error ? [...errors, result.error] : errors))(
              schema.safeParse(x)
            ),
          []
        );
        if (schemas.length - errors.length !== 1) {
          ctx.addIssue({
            path: ctx.path,
            code: "invalid_union",
            unionErrors: errors,
            message: "Invalid input: Should pass single schema",
          });
        }
      })
      .describe(
        "Length scale for the feature map, can be a single value or a list."
      )
      .optional(),
    num_frequencies: z.coerce
      .number()
      .int()
      .gte(1)
      .describe("Number of frequencies to use in the feature map.")
      .default(4),
    oversample_factor: z
      .number()
      .describe("Factor by which to oversample the feature map.")
      .optional(),
    num_oversample: z
      .number()
      .int()
      .describe("Number of oversamples to use, -1 for no oversampling.")
      .optional(),
    noise_scale: z
      .number()
      .gte(0)
      .describe("Scale of the noise to add to the system dynamics.")
      .optional(),
    plot: z.boolean().describe("Whether to plot the results.").optional(),
    verify: z.boolean().describe("Whether to verify the results.").optional(),
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
  .superRefine((data, ctx) => {
    const inputDimensions = data.X_bounds.length
      ? data.X_bounds[0].RectSet.length
      : 0;
    const valid =
      data.X_bounds.every((r) => r.RectSet.length === inputDimensions) &&
      data.X_init.every((r) => r.RectSet.length === inputDimensions) &&
      data.X_unsafe.every((r) => r.RectSet.length === inputDimensions);
    if (valid) return;
    for (const key of ["X_bounds", "X_init", "X_unsafe"] as const) {
      ctx.addIssue({
        path: [key],
        code: "custom",
        message: "X_bounds, X_init, and X_unsafe must have the same dimension.",
      });
    }
  })
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
    if (xs.size !== inputDimensions) {
      ctx.addIssue({
        path: ["system_dynamics"],
        code: "custom",
        message: `System dynamics must reference all and only input components [x1, ..., x${inputDimensions}].`,
      });
    }
    for (let i = 1; i <= xs.size; i++) {
      if (!xs.has(i)) {
        ctx.addIssue({
          path: ["system_dynamics"],
          code: "custom",
          message: `System dynamics must reference all and only input components [x1, ..., x${xs.size}].`,
        });
      }
    }
  })
  .superRefine((data, ctx) => {
    if (
      data.system_dynamics.length !== 0 ||
      (data.x_samples.length !== 0 && data.xp_samples.length !== 0)
    ) {
      return;
    }
    for (const cause of [
      "system_dynamics",
      "x_samples",
      "xp_samples",
    ] as const) {
      if (data[cause].length > 0) continue;
      ctx.addIssue({
        path: [cause],
        code: "custom",
        message: `Required`,
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
  .describe(
    "Representation of the command line arguments for pylucid expressed in a configuration file"
  );

export type Configuration = z.infer<typeof configurationSchema>;
