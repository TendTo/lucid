#!/usr/bin/env python3
import argparse

import pandas as pd
from benchmark import single_benchmark
from collect_results import config_from_df_row
from mlflow import MlflowClient

from pylucid import *
from pylucid import __version__
from pylucid.pipeline import pipeline


class BenchmarkArgs(argparse.Namespace):
    experiment: str
    verified: bool
    datafile: str
    tune: bool
    num_rows: int
    start_from: int
    config: bool
    dry_run: bool
    avoid_duplicates: bool


def check_exists(config: Configuration, args: BenchmarkArgs) -> bool:
    """Check if the given configuration already exists in the database."""
    client = MlflowClient(tracking_uri="http://localhost:5000")
    experiments = client.search_experiments(filter_string=f"name = '{args.experiment}-dist'")
    f = (
        f'metrics.run.safety > 0 and metrics.run.safety < 1 and metrics.run.success = 1 and params.seed = "{config.seed}"'
        f' and params.num_samples = "{config.num_samples}" and params.noise_scale = "{config.noise_scale}"'
        f' and params.estimator = "{config.estimator.__class__.__name__}" and params.feature_map = "{config.feature_map.__class__.__name__}"'
        f' and params.sigma_f = "{config.sigma_f}" and params.num_frequencies = "{config.num_frequencies}"'
        f' and params.lattice_resolution = "{config.lattice_resolution}" and params.feature_sigma_l = "{config.feature_sigma_l.tolist()}"'
        f' and params.set_scaling = "{config.set_scaling}" and params.sigma_l = "{config.sigma_l.tolist()}"'
    )
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in experiments],
        filter_string=f,
        order_by=["metrics.run.safety desc"],
    )
    print(f"Found {len(runs)}")
    return len(runs) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Experiment name to run")
    parser.add_argument("-v", "--verified", action="store_true", help="Only run the verified benchmarks")
    parser.add_argument(
        "-d",
        "--datafile",
        type=str,
        help="Path to the data file. Defaults to 'benchmarks/integration/<experiment>.pkl'",
        default="",
    )
    parser.add_argument("-n", "--num-rows", type=int, help="Number of rows to process", default=1000000)
    parser.add_argument("--start-from", type=int, help="Row index to start from", default=0)
    parser.add_argument("-t", "--tune", action="store_true", help="Run in tuning mode (if applicable)")
    parser.add_argument(
        "-c",
        "--config",
        action="store_true",
        help="Instead of running the experiment, save the configuration to a config file",
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run without executing benchmarks")
    parser.add_argument("-a", "--avoid-duplicates", action="store_true", help="Avoid running duplicate configurations")
    args: BenchmarkArgs = parser.parse_args()
    data: pd.DataFrame = pd.read_pickle(
        f"benchmarks/integration/{args.experiment}.pkl" if args.datafile == "" else args.datafile
    )
    data.sort_values(by=["obj_val"], ascending=True, inplace=True)
    if args.verified:
        data = data[data["verified"] == 1]
        print(f"Running only verified benchmarks: {len(data)} configurations found.")
    else:
        print(f"Running all benchmarks: {len(data)} configurations found.")
    if not args.dry_run:
        r = input("NOT DRY RUN. Type 'yes' to continue...")
        if r.lower() not in ("yes", "y", "1", "true", "ok"):
            exit(0)
    for i, (_, row) in enumerate(data.iterrows()):
        if hasattr(row, "sample_score") and row.sample_score < 0.9:
            print(f"Skipping row {i} due to low sample score: {row.sample_score}")
            args.num_rows += 1  # compensate for skipped rows
            continue
        if row.seed != 42:
            print(f"Skipping row {i} due to non-standard seed: {row.seed}")
            args.num_rows += 1  # compensate for skipped rows
            continue
        if i < args.start_from:
            continue
        if i >= args.start_from + args.num_rows:
            break
        row.num_samples = 1000
        config = config_from_df_row(args.experiment, row)
        if args.avoid_duplicates and check_exists(config, args):
            print(f"Skipping row {i} due to existing configuration in database.")
            continue
        if args.tune:
            config.estimator = config.estimator.__class__
        if args.config:
            config.estimator = config.estimator.__class__
            config.feature_map = config.feature_map.__class__
            config.to_yaml(f"{args.experiment}-config-{i}.yaml")
            continue
        if args.dry_run:
            config.estimator = config.estimator.__class__
            config.feature_map = config.feature_map.__class__
            print(f"Dry run for config {i}:")
            pipeline(
                config,
                optimiser_cb=lambda res: print(
                    f"Close enough: {row.obj_val}"
                    if np.isclose(res["obj_val"], row.obj_val)
                    else f"Missmatch: {res['obj_val']} vs {row.obj_val}"
                ),
            )
            continue
        single_benchmark(f"{args.experiment}-dist", config)
