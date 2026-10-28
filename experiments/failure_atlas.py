import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv
from atlas.runner import run_all, write_report

def run_failure_atlas_experiment(verbose=True):
    """Executes the 23-case Failure Atlas suite and outputs summary report."""
    os.makedirs("results/reports", exist_ok=True)
    os.makedirs("results/histories", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # 1. Run all cases (this also generates all 23 PNG plots and history CSVs)
    results = run_all(verbose=verbose)

    # 2. Write report
    report_path = write_report(results, "results/reports/failure_atlas.txt")

    # 3. Save aggregated summary CSV
    csv_path = "results/histories/failure_atlas.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "algorithm_number", "algorithm_name", "failure_title",
            "detected", "detail", "fail_iters", "fail_f", "good_iters", "good_f",
            "plot_path"
        ], extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    if verbose:
        print(f"Aggregated summary CSV: {csv_path}")
        print(f"Report: {report_path}")

    return results

run_failure_atlas = run_failure_atlas_experiment

if __name__ == "__main__":
    run_failure_atlas_experiment(verbose=True)
