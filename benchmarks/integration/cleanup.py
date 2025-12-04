#!/usr/bin/env python3
import shutil
from pathlib import Path
from typing import Optional


def get_metric_value(run_dir: Path, metric_name: str) -> Optional[float]:
    """
    Read a metric value from the metrics file.

    Args:
        run_dir: Path to the run directory
        metric_name: Name of the metric to read

    Returns:
        The metric value or None if not found
    """
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        return None

    metric_file = metrics_dir / metric_name
    if not metric_file.exists():
        return None

    try:
        # MLflow metrics files contain: timestamp value step
        # We want the last (most recent) value
        with open(metric_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                # Get last line and extract the value (second column)
                last_line = lines[-1].strip()
                parts = last_line.split()
                if len(parts) >= 2:
                    return float(parts[1])
    except (ValueError, IOError) as e:
        print(f"Error reading metric {metric_name} from {run_dir}: {e}")
        return None

    return None


def get_run_name(run_dir: Path) -> str:
    """
    Get the run name from meta.yaml if available.

    Args:
        run_dir: Path to the run directory

    Returns:
        The run name or the run ID
    """
    meta_file = run_dir / "meta.yaml"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Simple parsing for run_name
                for line in content.split("\n"):
                    if line.startswith("run_name:"):
                        return line.split(":", 1)[1].strip().strip("\"'")
        except IOError:
            pass

    return run_dir.name


def delete_low_safety_runs(
    mlruns_dir: str = "mlruns",
    experiment_id: str = "0",
    metric_name: str = "run.safety",
    threshold: float = 0.1,
    dry_run: bool = False,
    remove: bool = False,
) -> None:
    """
    Delete runs with safety metric below threshold.

    Args:
        mlruns_dir: Path to mlruns directory
        experiment_id: Experiment ID to process
        metric_name: Name of the safety metric
        threshold: Threshold below which runs are deleted
        dry_run: If True, only print what would be deleted without actually deleting
    """
    mlruns_path = Path(mlruns_dir)
    experiment_path = mlruns_path / experiment_id

    if not experiment_path.exists():
        print(f"Experiment directory not found: {experiment_path}")
        return

    print(f"Processing experiment {experiment_id} in {mlruns_dir}")
    print(f"Deleting runs with {metric_name} < {threshold}")
    print(f"{'DRY RUN - ' if dry_run else ''}Scanning runs...\n")

    deleted_count = 0
    kept_count = 0
    no_metric_count = 0

    # Iterate through all run directories in the experiment
    for run_dir in experiment_path.iterdir():
        if not run_dir.is_dir():
            continue

        # Skip meta.yaml and other non-run files
        if run_dir.name.startswith(".") or run_dir.name == "meta.yaml":
            continue

        run_id = run_dir.name
        run_name = get_run_name(run_dir)
        safety_value = get_metric_value(run_dir, metric_name)

        if safety_value is None:
            print(f"  ⚠️  Run {run_name} ({run_id[:8]}...): No {metric_name} metric found")
            no_metric_count += 1
            kept_count += 1
            continue

        if safety_value < threshold:
            print(f"  🗑️  Run {run_name} ({run_id[:8]}...): {metric_name}={safety_value:.6f} < {threshold}")

            if not dry_run:
                try:
                    if remove:
                        shutil.rmtree(run_dir)
                    else:
                        shutil.move(run_dir, run_dir.parent.parent.parent / "deleted" / experiment_id / run_dir.name)
                    # shutil.rmtree(run_dir)
                    print("      ✅ Deleted")
                except Exception as e:
                    print(f"      ❌ Error deleting: {e}")
                    kept_count += 1
                    continue
            else:
                print("      [Would delete in real run]")

            deleted_count += 1
        else:
            print(f"  ✓  Run {run_name} ({run_id[:8]}...): {metric_name}={safety_value:.6f} >= {threshold} - Kept")
            kept_count += 1

    print(f"\n{'DRY RUN - ' if dry_run else ''}Summary:")
    print(f"  Deleted: {deleted_count}")
    print(f"  Kept: {kept_count}")
    print(f"  No metric: {no_metric_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Delete MLflow runs with safety metric below threshold")
    parser.add_argument(
        "-i", "--mlruns-dir", type=str, default="mlruns", help="Path to mlruns directory (default: mlruns)"
    )
    parser.add_argument("-e", "--experiment-id", type=str, default="0", help="Experiment ID to process (default: 0)")
    parser.add_argument(
        "-m", "--metric-name", type=str, default="run.safety", help="Name of the safety metric (default: run.safety)"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.1, help="Threshold below which runs are deleted (default: 0.1)"
    )
    parser.add_argument(
        "-d", "--dry-run", action="store_true", help="Only print what would be deleted without actually deleting"
    )
    parser.add_argument(
        "-r", "--remove", action="store_true", help="Actually remove the runs instead of moving to deleted folder"
    )

    args = parser.parse_args()

    delete_low_safety_runs(
        mlruns_dir=args.mlruns_dir,
        experiment_id=args.experiment_id,
        metric_name=args.metric_name,
        threshold=args.threshold,
        dry_run=args.dry_run,
        remove=args.remove,
    )
