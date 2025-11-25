import { compile } from "mathjs";
import type { Configuration } from "./schema";

export function parseSystemDynamics(
  funStr: string[]
): (x: Float64Array, rows: number, cols: number) => void {
  // Compile each expression using math.js
  const expressions = funStr.map((expr: string) => compile(expr));

  return (x: Float64Array, rows: number, cols: number) => {
    for (let i = 0; i < rows; i++) {
      // Build variable mapping for current row
      const scope: Record<string, number> = {};
      for (let j = 0; j < cols; j++) {
        scope[`x${j + 1}`] = x[i * cols + j];
      }

      // Evaluate each expression and update the row
      for (let j = 0; j < cols; j++) {
        if (j < expressions.length) {
          x[i * cols + j] = expressions[j].evaluate(scope) as number;
        }
      }
    }
  };
}

export async function getCapabilities() {
  const jslucid = (await import(__JSLUCID_PATH__)).default;
  const {
    ALGLIB_BUILD,
    GUROBI_BUILD,
    HIGHS_BUILD,
    MATPLOTLIB_BUILD,
    SOPLEX_BUILD,
    PSOCPP_BUILD,
    OMP_BUILD,
    CUDA_BUILD,
  } = await jslucid();
  return {
    GUROBI: GUROBI_BUILD,
    ALGLIB: ALGLIB_BUILD,
    HIGHS: HIGHS_BUILD,
    MATPLOTLIB: MATPLOTLIB_BUILD,
    PLOT: MATPLOTLIB_BUILD,
    VERIFICATION: false,
    GUI: false,
    SOPLEX: SOPLEX_BUILD,
    PSOCPP: PSOCPP_BUILD,
    OMP: OMP_BUILD,
    CUDA: CUDA_BUILD,
  };
}

export async function runJslucid(
  config: Configuration,
  logCb: (msg: string) => void
) {
  const jslucid = (await import(__JSLUCID_PATH__)).default;
  const {
    Solver,
    MultiSet,
    SphereSet,
    RectSet,
    EllipseSet,
    pipeline,
    Matrix,
    log,
  } = await jslucid();

  function toOptimiser(optimiser: Configuration["optimiser"]) {
    switch (optimiser) {
      case "AlglibOptimiser":
        return Solver.Alglib;
      case "GurobiOptimiser":
        return Solver.Gurobi;
      case "HighsOptimiser":
        return Solver.HiGHS;
      case "SoplexOptimiser":
        return Solver.SOPLEX;
    }
  }

  function toSet(
    set:
      | Configuration["X_bounds"]
      | Configuration["X_init"]
      | Configuration["X_unsafe"]
      | Configuration["X_bounds"][number]
  ): RectSet | MultiSet | EllipseSet | SphereSet {
    if (Array.isArray(set)) {
      if (set.length === 0) {
        throw new Error("Empty set cannot be converted.");
      }
      return set.length > 1
        ? new MultiSet(set.map((s) => toSet(s)))
        : toSet(set[0]);
    }
    if ("SphereSet" in set) {
      return new SphereSet(set.SphereSet.center, set.SphereSet.radius);
    }
    if ("EllipseSet" in set) {
      return new EllipseSet(set.EllipseSet.center, set.EllipseSet.radii);
    }
    if ("RectSet" in set) {
      return new RectSet(set.RectSet);
    }
    throw new Error(`Unknown set type: ${JSON.stringify(set)}`);
  }

  log.set_verbosity(config.verbose);
  log.set_sink(logCb);

  return pipeline({
    verbose: config.verbose,
    system_dynamics: parseSystemDynamics(config.system_dynamics),
    X_bounds: toSet(config.X_bounds),
    X_init: toSet(config.X_init),
    X_unsafe: toSet(config.X_unsafe),
    f_xp_samples:
      config.x_samples && config.x_samples.length > 0
        ? new Matrix(config.x_samples)
        : null,
    xp_samples:
      config.x_samples && config.x_samples.length > 0
        ? new Matrix(config.x_samples)
        : null,
    x_samples:
      config.x_samples && config.x_samples.length > 0
        ? new Matrix(config.x_samples)
        : null,
    epsilon: config.epsilon,
    b_kappa: config.b_kappa,
    b_norm: config.b_norm,
    set_scaling: config.set_scaling,
    seed: config.seed,
    gamma: config.gamma,
    time_horizon: config.time_horizon,
    num_samples: config.num_samples,
    lambda: config.lambda,
    sigma_f: config.sigma_f,
    sigma_l: Array.isArray(config.sigma_l)
      ? config.sigma_l
      : Array(config.dimension).fill(config.sigma_l),
    feature_sigma_l: Array.isArray(config.feature_sigma_l)
      ? config.feature_sigma_l
      : Array(config.dimension).fill(config.feature_sigma_l),
    num_frequencies: config.num_frequencies,
    C_coeff: config.C_coeff,
    plot: config.plot,
    verify: config.verify,
    problem_log_file: config.problem_log_file,
    iis_log_file: config.iis_log_file,
    oversample_factor: config.oversample_factor,
    lattice_resolution: config.lattice_resolution,
    noise_scale: config.noise_scale,
    solver: toOptimiser(config.optimiser),
  });
}
