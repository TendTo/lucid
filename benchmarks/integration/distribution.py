#!/usr/bin/env python3
import argparse

from benchmark import single_benchmark
import pandas as pd
from collect_results import config_from_df_row

from pylucid import *
from pylucid import __version__

class BenchmarkArgs(argparse.Namespace):
    experiment: str
    verified: bool
    seeds: list[int] | None
    tune: bool
    num_rows: int
    start_from: int

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Experiment name to run")
    parser.add_argument("-v", "--verified", action="store_true", help="Only run the verified benchmarks")
    parser.add_argument("-s", "--seeds", type=int, nargs="+", help="Specific seeds to run (overrides CSV data)")
    parser.add_argument("-n", "--num-rows", type=int, help="Number of rows to process", default=-1)
    parser.add_argument("--start-from", type=int, help="Row index to start from", default=0)
    parser.add_argument("-t", "--tune", action="store_true", help="Run in tuning mode (if applicable)")
    args: BenchmarkArgs = parser.parse_args()
    data: pd.DataFrame = pd.read_pickle(f"benchmarks/integration/{args.experiment}.pkl")
    if args.verified:
        data = data[data['verified'] == 1]
        print(f"Running only verified benchmarks: {len(data)} configurations found.")
    for i, row in enumerate(data.itertuples()):
        if i < args.start_from:
            continue
        if args.num_rows != -1 and i >= args.start_from + args.num_rows:
            break
        config = config_from_df_row(args.experiment, row)
        for seed in (args.seeds if args.seeds is not None else [row.seed]):
            config.seed = seed
        if args.tune:
            config.estimator = KernelRidgeRegressor
        single_benchmark(f"{args.experiment}-dist", config)
