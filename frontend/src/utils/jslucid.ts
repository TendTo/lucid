import type { Configuration } from "./schema";
import type {
  RectSet as RectSetObj,
  EllipseSet as EllipseSetObj,
  SphereSet as SphereSetObj,
  MultiSet as MultiSetObj,
} from "./plot";

const jslucid_path = __JSLUCID_PATH__;

async function parseSystemDynamics(
  funStr: string[]
): Promise<(x: Float64Array, rows: number, cols: number) => void> {
  const { compile } = await import("mathjs");
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

function toSetObject(
  set:
    | Configuration["X_bounds"]
    | Configuration["X_init"]
    | Configuration["X_unsafe"]
    | Configuration["X_bounds"][number]
): RectSetObj | MultiSetObj | EllipseSetObj | SphereSetObj {
  if (Array.isArray(set)) {
    if (set.length === 0) {
      throw new Error("Empty set cannot be converted.");
    }
    return set.length > 1
      ? ({
          sets: set.map((s) => toSetObject(s)),
          dimension: toSetObject(set[0]).dimension,
        } as MultiSetObj)
      : toSetObject(set[0]);
  }
  if ("SphereSet" in set) {
    return {
      center: set.SphereSet.center,
      radius: set.SphereSet.radius,
      dimension: set.SphereSet.center.length,
    } as SphereSetObj;
  }
  if ("EllipseSet" in set) {
    return {
      center: set.EllipseSet.center,
      radii: set.EllipseSet.radii,
      dimension: set.EllipseSet.center.length,
    } as EllipseSetObj;
  }
  if ("RectSet" in set) {
    return {
      dimension: set.RectSet.length,
      lower_bound: set.RectSet.map((b: [number, number]) => b[0]),
      upper_bound: set.RectSet.map((b: [number, number]) => b[1]),
    } as RectSetObj;
  }
  throw new Error(`Unknown set type: ${JSON.stringify(set)}`);
}

export async function getCapabilities() {
  const jslucid = (await import(jslucid_path)).default;
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

async function toJslucidConfiguration(
  config: Configuration,
  num_points_plot: number,
  Solver: any,
  MultiSet: any,
  SphereSet: any,
  RectSet: any,
  EllipseSet: any
) {
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
  ): typeof RectSet | typeof MultiSet | typeof EllipseSet | typeof SphereSet {
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

  return {
    verbose: config.verbose,
    system_dynamics: await parseSystemDynamics(config.system_dynamics),
    X_bounds: toSet(config.X_bounds),
    X_init: toSet(config.X_init),
    X_unsafe: toSet(config.X_unsafe),
    f_xp_samples: config.x_samples,
    xp_samples: config.x_samples,
    x_samples: config.x_samples,
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
    num_points_plot: num_points_plot,
    problem_log_file: config.problem_log_file,
    iis_log_file: config.iis_log_file,
    oversample_factor: config.oversample_factor,
    lattice_resolution: config.lattice_resolution,
    noise_scale: config.noise_scale,
    solver: toOptimiser(config.optimiser),
  };
}

export async function generatePreviewFigure(config: Configuration) {
  const plotData = (await import("./plot")).plotData;
  if (config.x_samples.length > 0) {
    return plotData(
      { x_lattice: config.x_samples, xp_lattice: config.xp_samples },
      toSetObject(config.X_bounds) as RectSetObj,
      {
        XInit: toSetObject(config.X_init),
        XUnsafe: toSetObject(config.X_unsafe),
      }
    );
  }

  const plotFunction = (await import("./plot")).plotFunction;
  const jslucid = (await import(jslucid_path)).default;
  const { plot_preview, Solver, MultiSet, SphereSet, RectSet, EllipseSet } =
    await jslucid();
  const jsConfig = await toJslucidConfiguration(
    config,
    config.num_samples,
    Solver,
    MultiSet,
    SphereSet,
    RectSet,
    EllipseSet
  );
  jsConfig.num_samples = Math.min(50, jsConfig.num_samples);
  const previewData = plot_preview(jsConfig);
  return plotFunction(previewData, toSetObject(config.X_bounds) as RectSetObj, {
    XInit: toSetObject(config.X_init),
    XUnsafe: toSetObject(config.X_unsafe),
  });
}

export async function runJslucid(
  config: Configuration,
  logCb: (msg: string) => void
) {
  const jslucid = (await import(jslucid_path)).default;
  const { pipeline, log, Solver, MultiSet, SphereSet, RectSet, EllipseSet } =
    await jslucid();

  log.set_verbosity(config.verbose);
  log.set_sink(logCb);

  // const [solution, plotData] = pipeline(await toJslucidConfiguration(config));
  const [solution, plotData] = pipeline(
    await toJslucidConfiguration(
      config,
      100,
      Solver,
      MultiSet,
      SphereSet,
      RectSet,
      EllipseSet
    )
  );

  if (solution.success) {
    const { plotSolution } = await import("./plot");
    const fig = plotSolution(
      plotData,
      toSetObject(config.X_bounds) as RectSetObj,
      {
        c: solution.c,
        gamma: config.gamma,
        eta: solution.eta,
        XInit: toSetObject(config.X_init),
        XUnsafe: toSetObject(config.X_unsafe),
      }
    );
    return { ...solution, fig };
  }
  return { ...solution, fig: undefined };
}
