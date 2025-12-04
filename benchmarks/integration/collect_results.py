import argparse
import json
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Run
from plot_solution import (
    base_load_configuration,
    plot_contour_benchmarks,
    plot_solution_matplotlib,
    load_configuration,
)

from pylucid import ModelEstimator

FILTER = "metrics.run.safety > 0 and metrics.run.safety < 1 and metrics.run.success = 1"


@dataclass
class Args(argparse.Namespace):
    experiment: str
    points: int
    elevation: float
    azimuth: float
    roll: float
    verify: bool
    plot_bxp: bool
    plot_bxe: bool
    uri: str
    d_uri: str
    download: bool
    plot: bool
    filter: str
    output: str
    to_config: bool


def plot_solution(args: Args, data: pd.DataFrame):
    config = base_load_configuration(f"benchmarks/integration/{args.experiment.lower()}.yaml")
    if isinstance(data, tuple):
        data = pd.DataFrame([data._asdict()])
    for run in data.itertuples():
        feature_map = config.feature_map(
            num_frequencies=run.num_frequencies,
            sigma_l=run.feature_sigma_l,
            sigma_f=run.sigma_f,
            X_bounds=config.X_bounds,
        )
        if config.estimator != ModelEstimator:
            estimator = config.estimator(
                kernel=config.kernel(sigma_l=run.sigma_l, sigma_f=run.sigma_f),
                regularization_constant=run.lambda_,
            )
        else:
            estimator = config.estimator(lambda x: feature_map(config.system_dynamics(x)))
        estimator.consolidate(config.x_samples, feature_map(config.xp_samples))
        plot_solution_matplotlib(
            args=args,
            c=run.c,
            eta=run.eta,
            estimator=estimator,
            f=config.system_dynamics,
            feature_map=feature_map,
            gamma=run.gamma,
            sol=run.solution,
            X_bounds=config.X_bounds,
            X_init=config.X_init,
            X_unsafe=config.X_unsafe,
            num_samples=args.points,
        )


def get_bounds(bounds: "MultiSet | RectSet") -> tuple[list[np.ndarray], list[np.ndarray]]:
    if isinstance(bounds, RectSet):
        return [bounds.lower_bound], [bounds.upper_bound]
    if isinstance(bounds, MultiSet):
        lb, ub = np.array([]), np.array([])
        for s in bounds:
            if isinstance(s, RectSet):
                lb = np.vstack([lb, s.lower_bound]) if lb.size else s.lower_bound
                ub = np.vstack([ub, s.upper_bound]) if ub.size else s.upper_bound
        return [lb], [ub]
    raise TypeError("Unsupported bounds type")


def export_solution(args: Args, data: pd.DataFrame) -> pd.DataFrame:
    config = base_load_configuration(f"benchmarks/integration/{args.experiment.lower()}.yaml")
    if isinstance(data, tuple):
        data = pd.DataFrame([data._asdict()])
    for run in data.itertuples():
        feature_map = config.feature_map(
            num_frequencies=run.num_frequencies,
            sigma_l=run.sigma_l,
            sigma_f=run.sigma_f,
            X_bounds=config.X_bounds,
        )
        estimator = config.estimator(
            kernel=config.kernel(sigma_l=run.sigma_l, sigma_f=run.sigma_f),
            regularization_constant=run.lambda_,
        )
        estimator.consolidate(config.x_samples, feature_map(config.xp_samples))

        data = data.copy()
        x_lattice = config.X_bounds.lattice(config.num_samples or 1000, True)
        assert isinstance(config.X_bounds, RectSet)

        # data["X_bounds_lower"], data["X_bounds_upper"] = get_bounds(config.X_bounds)
        # data["X_init_lower"], data["X_init_upper"] = get_bounds(config.X_init)
        # data["X_unsafe_lower"], data["X_unsafe_upper"] = get_bounds(config.X_unsafe)

        data["x_lattice"] = x_lattice
        data["x_barrier_values"] = feature_map(x_lattice) @ run.solution.T
        data["xp_est_barrier_values"] = estimator(x_lattice) @ run.solution.T
        if config.system_dynamics:
            data["xp_barrier_values"] = feature_map(config.system_dynamics(x_lattice)) @ run.solution.T

        return data


def get_solution(run: "Run", d_uri: str):
    _, path = run.info.artifact_uri.split("/mlruns/")
    # Get the path of this python script
    #
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "mlruns", path, "solution.json")
    print(f"Loading solution from {file_path} ...")
    if not os.path.exists(file_path):
        print(f"WARNING: File {file_path} does not exist.")
        return np.array([])
    with open(file_path, "r") as f:
        content = json.load(f)
    if "data" not in content:
        print(f"WARNING: File {file_path} does not contain 'data' key.")
        return np.array([])
    return np.array(content["data"]).flatten()


def float_or_array(run: "Run", key: str) -> float | np.ndarray:
    value = run.data.params[key]
    try:
        # Try to evaluate as a Python literal (e.g., list, array)
        evaluated_value = eval(value)
        if isinstance(evaluated_value, (list, np.ndarray)):
            return np.array(evaluated_value)
        return float(evaluated_value)
    except (SyntaxError, NameError):
        # If eval fails, treat it as a float
        return float(value)


def get_data_from_mlflow(args: Args):
    # Create an experiment with a name that is unique and case sensitive.
    client = MlflowClient(tracking_uri=args.uri)
    experiments = client.search_experiments(filter_string=f"name = '{args.experiment}'")
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in experiments],
        filter_string=args.filter,
        order_by=["metrics.run.obj_val asc"],
    )
    print(f"Found {len(runs)} runs in experiment '{args.experiment}'.")
    data = pd.DataFrame(
        {
            # Params
            "seed": int(run.data.params["seed"]),
            "sigma_f": float(run.data.params["sigma_f"]),
            "sigma_l": (
                float_or_array(run, "fit.sigma_l")
                if "fit.sigma_l" in run.data.params
                else float_or_array(run, "sigma_l")
            ),
            "lambda_": float(run.data.params.get("lambda", None) or run.data.params.get("lambda_", None)),
            "num_frequencies": int(run.data.params["num_frequencies"]),
            "lattice_resolution": int(
                (
                    run.data.params["lattice_resolution"]
                    if run.data.params["lattice_resolution"] != "-1"
                    else np.ceil(
                        (2 * int(run.data.params["num_frequencies"]) + 1) * float(run.data.params["oversample_factor"])
                    )
                ),
            ),
            "T": int(run.data.params["time_horizon"]),
            "gamma": float(run.data.params["gamma"]),
            "noise_scale": float(run.data.params["noise_scale"]),
            "set_scaling": float(run.data.params["set_scaling"]),
            "num_samples": int(run.data.params["num_samples"]),
            "feature_sigma_l": float_or_array(run, "feature_sigma_l"),
            "oversample_factor": float(run.data.params["oversample_factor"]),
            "b_kappa": float(run.data.params.get("b_kappa", 1.0)),
            "b_norm": float(run.data.params.get("b_norm", 1.0)),
            "C_coeff": float(run.data.params.get("C_coeff", 1.0)),
            "epsilon": float(run.data.params.get("epsilon", 0.0)),
            # Metrics
            "eta": float(run.data.metrics["run.eta"]),
            "c": float(run.data.metrics["run.c"]),
            "norm": float(run.data.metrics["run.norm"]),
            "obj_val": 1 - float(run.data.metrics["run.safety"]),
            "percentage": float(run.data.metrics["run.safety"]) * 100,
            # Format time as MM:SS
            "time_milliseconds": run.info.end_time - run.info.start_time,
            "time": f"{(run.info.end_time - run.info.start_time) // 1000 // 60}:{(run.info.end_time - run.info.start_time) // 1000 % 60:02d}",
            "peak_rss_memory_usage_bytes": float(run.data.metrics.get("run.peak_rss_memory_usage_bytes", -1)),
            "num_variables": int(run.data.metrics.get("run.num_variables", -1)),
            "num_constraints": int(run.data.metrics.get("run.num_constraints", -1)),
            # Results
            "solution": get_solution(run, args.d_uri),
            "verified": -1,
        }
        for run in runs
    )
    data.to_pickle(f"benchmarks/integration/{args.experiment.lower()}.pkl")
    return data


def get_data_from_pickle(args: Args):
    """
    Load data from a pickle file.
    This is useful for testing or when the data is already available in a local file.
    """
    data = pd.read_pickle(f"benchmarks/integration/{args.experiment.lower()}.pkl")
    print(f"Loaded {len(data)} runs from pickle file for experiment '{args.experiment}'.")
    return data


LATEX_KEEPS = {
    "num_frequencies": "Freq.",
    "lattice_resolution": "Lattice Size",
    "feature_sigma_l": r"$\sigma_{l_f}$",
    "sigma_l": r"$\sigma_l$",
    "set_scaling": "Set Scale",
    "eta": r"$\eta$",
    "c": r"$c$",
    "time": "Runtime",
    "percentage": "Safety Prob.",
}


def print_latex_table(data: pd.DataFrame, experiment: str):
    latex_data = data[LATEX_KEEPS.keys()].sort_values(by=["percentage", "c"], ascending=[False, True])
    latex_data.percentage = latex_data.percentage.apply(lambda x: f"{x:.2f}\\%")
    latex_data.set_scaling = latex_data.set_scaling.apply(lambda x: f"{x * 100:.0f}\\%")
    latex_data.feature_sigma_l = latex_data.feature_sigma_l.apply(
        lambda x: "[" + ", ".join([f"{v:.2f}" for v in x]) + "]"
    )
    latex_data.rename(LATEX_KEEPS, axis=1).to_latex(
        f"benchmarks/integration/{experiment.lower()}.tex",
        index=False,
        float_format="%.2f",
        column_format="c" * len(LATEX_KEEPS),
    )


def config_from_df_row(experiment: str, row: pd.Series):
    config = base_load_configuration(f"benchmarks/integration/{experiment.lower()}.yaml", row.seed)
    config.seed = row.seed
    config.sigma_f = row.sigma_f
    config.sigma_l = row.sigma_l
    config.lambda_ = row.lambda_
    config.num_frequencies = row.num_frequencies
    config.lattice_resolution = row.lattice_resolution
    config.time_horizon = row["T"] if isinstance(row, pd.Series) else row.T
    config.gamma = row.gamma
    config.noise_scale = row.noise_scale
    config.set_scaling = row.set_scaling
    config.feature_sigma_l = row.feature_sigma_l
    config.num_samples = row.num_samples
    config.oversample_factor = row.oversample_factor
    config.b_kappa = row.b_kappa
    config.b_norm = row.b_norm
    config.C_coeff = row.C_coeff
    config.epsilon = row.epsilon
    config = load_configuration(config)
    return config


def main(args: Args):
    # Create an experiment with a name that is unique and case sensitive.
    data = get_data_from_mlflow(args) if args.download else get_data_from_pickle(args)
    data.sort_values(by=["obj_val"], ascending=True, inplace=True)
    if args.verify:
        from pylucid.dreal import verify_barrier_conditions

        verified_rows = data[data["verified"] == 1].shape[0]
        for i, row in data.iterrows():
            if verified_rows >= 10:
                break
            config = config_from_df_row(args.experiment, row)
            success = verify_barrier_conditions(
                X_bounds=config.X_bounds,
                X_init=config.X_init,
                X_unsafe=config.X_unsafe,
                estimator=config.estimator,
                b_norm=config.b_norm,
                c=row.c,
                eta=row.eta,
                gamma=row.gamma,
                epsilon=config.epsilon,
                sigma_f=config.sigma_f,
                sol=row.solution,
                tffm=config.feature_map,
            )
            if success:
                verified_rows += 1
                data.at[i, "verified"] = 1
            else:
                data.at[i, "verified"] = 0
            data.to_pickle(f"benchmarks/integration/{args.experiment.lower()}.pkl")
        data = data[data["verified"] == 1]

    # Remove duplicate runs based on the 'objective value' column
    data = data.drop_duplicates(subset=["obj_val"], keep="first")
    print(f"Found {len(data)} unique runs in experiment '{args.experiment}'.")
    print_latex_table(data, args.experiment)
    for i, row in enumerate(data.itertuples()):
        print(
            f"Experiment {args.experiment} took {row.time} ms\nSuccess: {row.percentage:.2f}%, c {row.c}, eta {row.eta}, lambda {row.lambda_}, num_frequencies {row.num_frequencies}, lattice_resolution {row.lattice_resolution}, oversample_factor {row.oversample_factor}, sigma_l {row.sigma_l}, sigma_f {row.sigma_f}, T {row.T}"
        )
        if args.output:
            data = export_solution(args, data)
            data.to_hdf(f"{args.output}-{i}.h5", mode="w")
            print(f"Exported solution to {args.output}-{i}.h5")
        if args.plot:
            r = input(f"Run {row.Index} - Print?...")
            if r.lower() == "y" or r.lower() == "yes":
                plot_solution(args, row)
        if args.to_config:
            # Ensure the output directory exists
            os.makedirs(f"benchmarks/integration/{args.experiment.lower()}", exist_ok=True)
            config = config_from_df_row(args, row)
            config.to_yaml(f"benchmarks/integration/{args.experiment.lower()}/{i}.yaml")
        print("---" * 20)
    plot_contour_benchmarks(
        args.experiment, x=data["num_frequencies"].values, y=data["lattice_resolution"].values, z=data["obj_val"].values
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect results from MLflow and plot them.",
    )
    parser.add_argument("experiment", type=str, help="Name of the MLflow experiment to collect results from.")
    parser.add_argument(
        "-u", "--uri", type=str, default="http://localhost:5000", help="URI of the MLflow tracking server."
    )
    parser.add_argument(
        "-d", "--d_uri", type=str, default="http://localhost:8000", help="URI of the MLflow download server."
    )
    parser.add_argument("-p", "--points", type=int, help="The number of points for the plot.", default=200)
    parser.add_argument("-e", "--elevation", type=float, help="The elevation angle for the plot.", default=30)
    parser.add_argument("-a", "--azimuth", type=float, help="The azimuth angle for the plot.", default=-15)
    parser.add_argument("-r", "--roll", type=float, help="The roll angle for the plot.", default=0)
    parser.add_argument("-v", "--verify", action="store_true", help="Verify the barrier certificate.")
    parser.add_argument("--download", action="store_true", help="Download data from MLflow.")
    parser.add_argument("--plot_bxp", action="store_true", help="Plot the B(xp) surface.")
    parser.add_argument("--plot_bxe", action="store_true", help="Plot the B(xp) est. surface.")
    parser.add_argument("--plot", action="store_true", help="Plot the solution.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output file path if the solution is to be exported, without the extension.",
    )
    parser.add_argument("-f", "--filter", type=str, default=FILTER, help="Filter for the MLflow runs.")
    parser.add_argument("--to-config", action="store_true", help="Export the configuration corresponding to each run.")
    main(parser.parse_args())
