import { z } from "zod";

export const jsonSchema = z
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
    system_dynamics: z
      .array(z.string().nonempty())
      .min(1)
      .describe("Dynamics of the system, describing how it evolves over time.")
      .optional(),
    X_bounds: z
      .object({
        RectSet: z
          .array(
            z
              .tuple([z.number(), z.number()])
              .refinement(([min, max]) => min < max, {
                message:
                  "Invalid bounds: lower bound must be less than upper bound.",
                code: "custom",
              })
          )
          .min(1)
          .describe(
            "Bounds for the state space, each tuple is a [lower, upper] bound."
          )
          .default([]),
      })
      .describe("Bounds for the state space.")
      .optional(),
    X_init: z
      .object({
        RectSet: z
          .array(
            z
              .tuple([z.number(), z.number()])
              .refinement(([min, max]) => min < max, {
                message:
                  "Invalid bounds: lower bound must be less than upper bound.",
                code: "custom",
              })
          )
          .min(1)
          .describe(
            "Initial state of the system, each tuple is a [lower, upper] bound."
          )
          .default([]),
      })
      .describe("Initial state of the system.")
      .optional(),
    X_unsafe: z
      .object({
        RectSet: z
          .array(
            z
              .tuple([z.number(), z.number()])
              .refinement(([min, max]) => min < max, {
                message:
                  "Invalid bounds: lower bound must be less than upper bound.",
                code: "custom",
              })
          )
          .min(1)
          .describe(
            "Unsafe states of the system, each tuple is a [lower, upper] bound."
          )
          .default([]),
      })
      .describe("Unsafe states of the system.")
      .optional(),
    gamma: z.number().optional(),
    c_coefficient: z.number().optional(),
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
    num_frequencies: z
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
      .enum(["GurobiOptimiser", "AlglibOptimiser"])
      .describe("Type of optimiser to use for the estimator.")
      .default("GurobiOptimiser"),
  })
  .strict()
  .superRefine((data, ctx) => {
    const valid =
      data.X_bounds?.RectSet !== undefined &&
      data.X_bounds.RectSet.length === data.X_init?.RectSet.length &&
      data.X_bounds.RectSet.length === data.X_unsafe?.RectSet.length;
    if (valid) return;
    for (const key of [
      "X_bounds.RectSet",
      "X_init.RectSet",
      "X_unsafe.RectSet",
    ] as const) {
      ctx.addIssue({
        path: [key],
        code: "custom",
        message: "X_bounds, X_init, and X_unsafe must have the same dimension.",
      });
    }
  })
  .superRefine((data, ctx) => {
    const xs = new Set();
    for (const f of data.system_dynamics ?? []) {
      const match = f.matchAll(/x(\d+)/g);
      console.log("match", match);
      for (const m of match ?? []) {
        const x = parseInt(m[1], 10);
        xs.add(x);
      }
    }
    console.log("xs", xs);
    if (xs.size !== data.X_bounds?.RectSet.length) {
      ctx.addIssue({
        path: ["system_dynamics"],
        code: "custom",
        message: `System dynamics must reference all and only inputs from 'x1' to 'x${data.X_bounds?.RectSet.length}'.`,
      });
    }
    for (let i = 1; i <= xs.size; i++) {
      if (!xs.has(i)) {
        ctx.addIssue({
          path: ["system_dynamics"],
          code: "custom",
          message: `System dynamics must reference all and only inputs from 'x1' to 'x${xs.size}'.`,
        });
      }
    }
  })
  .describe(
    "Representation of the command line arguments for pylucid expressed in a configuration file"
  );
