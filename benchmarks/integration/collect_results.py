import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from mlflow import MlflowClient, artifacts
from mlflow.entities import Run
from plot_solution import base_load_configuration, plot_solution_matplotlib

FILTER = 'params.c_coefficient = "1.0" and metrics.run.obj_val > 0 and metrics.run.obj_val < 1 and params.constant_lattice_points = "False" and metrics.run.success = 1'


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


def plot_solution(args: Args, data: pd.DataFrame):
    config = base_load_configuration(f"benchmarks/integration/{args.experiment.lower()}.yaml")
    if isinstance(data, tuple):
        data = pd.DataFrame([data._asdict()])
    for run in data.itertuples():
        feature_map = config.feature_map(
            num_frequencies=run.num_frequencies,
            sigma_l=run.sigma_l,
            sigma_f=run.sigma_f,
            x_limits=config.X_bounds,
        )
        estimator = config.estimator(
            kernel=config.kernel(sigma_l=run.sigma_l, sigma_f=run.sigma_f),
            regularization_constant=run.lambda_,
        )
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


def get_solution(run: "Run", d_uri: str):
    _, path = run.info.artifact_uri.split("/mlruns/")
    file = requests.get(f"{d_uri}/{path}/solution.json", timeout=10)
    if file.status_code == 200:
        file = file.json()
        return np.array(file["data"]).flatten()
    return np.array([])


def main(args: Args):
    # Create an experiment with a name that is unique and case sensitive.
    client = MlflowClient(tracking_uri=args.uri)
    experiments = client.search_experiments(filter_string=f"name = '{args.experiment}'")
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in experiments],
        filter_string=FILTER,
        order_by=["metrics.run.obj_val asc"],
    )
    print(f"Found {len(runs)} runs in experiment '{args.experiment}'.")
    data = pd.DataFrame(
        {
            # Params
            "sigma_f": float(run.data.params["sigma_f"]),
            "sigma_l": np.array(eval(run.data.params["sigma_l"])),
            "oversample_factor": float(run.data.params["oversample_factor"]),
            "lambda_": float(run.data.params["lambda_"]),
            "num_frequencies": int(run.data.params["num_frequencies"]),
            "num_oversample": int(
                (
                    run.data.params["num_oversample"]
                    if run.data.params["num_oversample"] != "-1"
                    else np.ceil(
                        (2 * int(run.data.params["num_frequencies"]) + 1) * float(run.data.params["oversample_factor"])
                    )
                ),
            ),
            "T": int(run.data.params["time_horizon"]),
            "gamma": float(run.data.params["gamma"]),
            "noise_scale": float(run.data.params["noise_scale"]),
            "oversample_factor": float(run.data.params["oversample_factor"]),
            # Metrics
            "eta": float(run.data.metrics["run.eta"]),
            "c": float(run.data.metrics["run.c"]),
            "norm": float(run.data.metrics["run.norm"]),
            "obj_val": float(run.data.metrics["run.obj_val"]),
            "percentage": (1 - float(run.data.metrics["run.obj_val"])) * 100,
            # Format time as MM:SS
            "time": f"{(run.info.end_time - run.info.start_time) // 1000 // 60}:{(run.info.end_time - run.info.start_time) // 1000 % 60:02d}",
            # Results
            "solution": get_solution(run, args.d_uri),
        }
        for run in runs
    )
    data.to_latex(
        f"benchmarks/integration/{args.experiment.lower()}.tex",
        index=False,
        columns=[
            "sigma_l",
            "sigma_f",
            "lambda_",
            "num_frequencies",
            "num_oversample",
            "eta",
            "gamma",
            "c",
            "T",
            "time",
            "percentage",
        ],
    )
    for i, row in enumerate(data.itertuples()):
        print(
            f"Experiment {args.experiment} took {row.time} ms\nSuccess: {row.percentage:.2f}%, c {row.c}, eta {row.eta}, lambda {row.lambda_}, num_frequencies {row.num_frequencies}, num_oversample {row.num_oversample}, oversample_factor {row.oversample_factor}, sigma_l {row.sigma_l}, sigma_f {row.sigma_f}, T {row.T}"
        )
        r = input(f"Run {row.Index} - Print?...")
        if r.lower() == "y" or r.lower() == "yes":
            plot_solution(args, row)
        print("---" * 20)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect results from MLflow and plot them.",
    )
    parser.add_argument("experiment", type=str, help="Name of the MLflow experiment to collect results from.")
    parser.add_argument(
        "-u", "--uri", type=str, default="http://localhost:5000", help="URI of the MLflow tracking server."
    )
    parser.add_argument(
        "-d", "--d_uri", type=str, default="http://localhost:8080", help="URI of the MLflow download server."
    )
    parser.add_argument("-p", "--points", type=int, help="The number of points for the plot.", default=200)
    parser.add_argument("-e", "--elevation", type=float, help="The elevation angle for the plot.", default=30)
    parser.add_argument("-a", "--azimuth", type=float, help="The azimuth angle for the plot.", default=-15)
    parser.add_argument("-r", "--roll", type=float, help="The roll angle for the plot.", default=0)
    parser.add_argument("-v", "--verify", action="store_true", help="Verify the barrier certificate.")
    parser.add_argument("--plot_bxp", action="store_true", help="Plot the B(xp) surface.")
    parser.add_argument("--plot_bxe", action="store_true", help="Plot the B(xp) est. surface.")
    main(parser.parse_args())

# Forwarding                    https://75002213d6dc.ngrok-free.app -> http://localhost:8080
# Forwarding                    https://c65058fe26b2.ngrok-free.app -> http://localhost:5000
# py .\benchmarks\integration\collect_results.py -u https://c65058fe26b2.ngrok-free.app -d https://75002213d6dc.ngrok-free.app Barrier2
