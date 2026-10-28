import os
import sys
import argparse
import unittest
from experiments.compare_algorithms import run_compare_algorithms
from experiments.regression_experiments import run_regression_experiments
from experiments.newton_quadratic import run_newton_quadratic_experiment
from experiments.constrained_experiments import run_constrained_experiments
from experiments.failure_atlas import run_failure_atlas
from experiments.all_algorithm_contours import (
    _gen_gd_family_contours, _gen_momentum_adaptive_contours,
    _gen_regression_contours, _gen_newton_bfgs_contours,
    _gen_constrained_contours, generate_all_algorithm_contours
)

def run_gd_family():
    """Executes comparison and generates contours for gradient descent variants."""
    print("\n--- [1] Running Gradient Descent Family (Algorithms 1-5) ---")
    run_compare_algorithms(verbose=True)
    _gen_gd_family_contours()
    print("Done GD Family.")

def run_momentum_nesterov():
    """Executes comparison and generates contours for Momentum and Nesterov."""
    print("\n--- [2] Running Momentum & Nesterov Acceleration (Algorithms 6-7) ---")
    run_compare_algorithms(verbose=True)
    _gen_momentum_adaptive_contours()
    print("Done Momentum & Nesterov.")

def run_regression_adaptive():
    """Executes regression and adaptive optimizer experiment pipelines."""
    print("\n--- [3] Running Regression & Adaptive Optimizers (Algorithms 8-14) ---")
    run_regression_experiments(verbose=True)
    _gen_regression_contours()
    _gen_momentum_adaptive_contours()
    print("Done Regression & Adaptive Optimizers.")

def run_newton_bfgs():
    """Executes Newton and BFGS experiments on quadratic and Rosenbrock."""
    print("\n--- [4] Running Newton & BFGS Quasi-Newton (Algorithms 17-20) ---")
    run_newton_quadratic_experiment(verbose=True)
    _gen_newton_bfgs_contours()
    print("Done Newton & BFGS.")

def run_nonsmooth_constrained():
    """Executes nonsmooth and constrained optimization experiments."""
    print("\n--- [5] Running Nonsmooth & Constrained Optimization (Algorithms 15-16, 21-23) ---")
    run_constrained_experiments(verbose=True)
    _gen_constrained_contours()
    print("Done Nonsmooth & Constrained Optimization.")

def run_atlas():
    """Executes the 23-case Failure Atlas pipeline and generates reports."""
    print("\n--- [6] Running 23-Case Failure Atlas ---")
    results = run_failure_atlas(verbose=True)
    flagged = sum(1 for r in results if r["detected"])
    print(f"Failure Atlas complete: {flagged}/{len(results)} failure modes detected and flagged.")
    return results

def run_everything():
    """Runs all 23 algorithms, all experiment suites, and Failure Atlas."""
    print("\n================================================================================")
    print("Running Full Suite: All 23 Optimization Algorithms & 23-Case Failure Atlas")
    print("================================================================================")
    run_compare_algorithms(verbose=True)
    run_regression_experiments(verbose=True)
    run_newton_quadratic_experiment(verbose=True)
    run_constrained_experiments(verbose=True)
    generate_all_algorithm_contours(verbose=True)
    run_failure_atlas(verbose=True)
    print("\nAll algorithms, experiments, and failure atlas cases completed successfully.")

def run_tests():
    """Discovers and runs the full automated unit test suite."""
    print("\n================================================================================")
    print("Running Unit Test Suite: Discovering tests in tests/...")
    print("================================================================================")
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"\nTest Summary: {passed}/{result.testsRun} tests passed (Errors={len(result.errors)}, Failures={len(result.failures)}).")
    return result.wasSuccessful()

def interactive_menu():
    """Displays the interactive console menu for experiment selection."""
    options = {
        "1": ("GD family", run_gd_family),
        "2": ("Momentum/Nesterov", run_momentum_nesterov),
        "3": ("Regression and adaptive", run_regression_adaptive),
        "4": ("Newton/BFGS", run_newton_bfgs),
        "5": ("Nonsmooth/Constrained", run_nonsmooth_constrained),
        "6": ("Failure Atlas", run_atlas),
        "7": ("Run everything", run_everything),
        "0": ("Exit", lambda: sys.exit(0))
    }
    while True:
        print("\n" + "=" * 60)
        print("OptiBreak Interactive Optimization Suite")
        print("=" * 60)
        for key in ["1", "2", "3", "4", "5", "6", "7", "0"]:
            print(f" {key}. {options[key][0]}")
        print("=" * 60)
        try:
            choice = input("Select an option (0-7): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting OptiBreak.")
            break
        if choice in options:
            if choice == "0":
                print("Exiting OptiBreak.")
                sys.exit(0)
            options[choice][1]()
        else:
            print(f"Invalid option '{choice}'. Please choose from 0 to 7.")

def main():
    """CLI entry point parsing command-line flags and launching suites."""
    parser = argparse.ArgumentParser(description="OptiBreak Optimization Suite & Failure Atlas")
    parser.add_argument("--all", action="store_true", help="Run all 23 algorithms, experiments, and Failure Atlas")
    parser.add_argument("--atlas", action="store_true", help="Run the 23-case Failure Atlas")
    parser.add_argument("--test", action="store_true", help="Run unittest test discovery")

    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)
    elif args.all:
        run_everything()
    elif args.atlas:
        run_atlas()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
