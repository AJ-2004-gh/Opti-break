import os
import unittest
from core.convergence import Result
from atlas.cases import CASES, ALL_CASES
from atlas.runner import run_case, run_all, write_report

class TestFailureAtlas(unittest.TestCase):
    def test_case_count_and_numbering(self):
        """
        Asserts there are exactly 23 cases and algorithm_number covers 1..23 with no gaps.
        """
        self.assertEqual(len(CASES), 23, "Failure Atlas must contain exactly 23 cases")
        alg_nums = [c["algorithm_number"] for c in CASES]
        self.assertEqual(sorted(alg_nums), list(range(1, 24)), "algorithm_number must cover 1..23 with no gaps")

    def test_all_cases_good_and_detector(self):
        """
        For every case: the good run succeeds (converges or reports its expected
        normal result) and the detector returns True on the bad run.
        """
        for case in CASES:
            case_id = case["id"]
            with self.subTest(case_id=case_id):
                # 1. Good run succeeds
                good_res = case["good_run"]()
                if isinstance(good_res, Result):
                    success = good_res.converged or (good_res.fun < 0.1)
                elif isinstance(good_res, dict):
                    success = (
                        good_res.get("converged", False) or
                        good_res.get("holds", False) or
                        good_res.get("kkt", False) or
                        (good_res.get("residual_dist", 1.0) < 1e-8)
                    )
                elif hasattr(good_res, "failed"):
                    success = not good_res.failed
                elif isinstance(good_res, bool):
                    success = good_res is True
                else:
                    success = True

                self.assertTrue(
                    success,
                    f"Case {case_id} ({case['algorithm_name']}): good_run did not succeed ({good_res})"
                )

                # 2. Detector returns False on good run
                try:
                    det_good, detail_good = case["detector"](good_res, good_res=good_res)
                except TypeError:
                    det_good, detail_good = case["detector"](good_res)
                self.assertFalse(
                    det_good,
                    f"Case {case_id} ({case['algorithm_name']}): detector returned True on good run ({detail_good})"
                )

                # 3. Detector flags bad run
                bad_res = case["bad_run"]()
                try:
                    detected, detail = case["detector"](bad_res, good_res=good_res)
                except TypeError:
                    detected, detail = case["detector"](bad_res)
                self.assertTrue(
                    detected,
                    f"Case {case_id} ({case['algorithm_name']}): detector returned False on bad run ({detail})"
                )

    def test_run_all_writes_artifacts(self):
        """
        Asserts run_all writes 23 PNG files and a report containing every algorithm name.
        """
        results = run_all(verbose=False)
        self.assertEqual(len(results), 23)

        # Check 23 PNG plot files exist
        for i in range(1, 24):
            png_path = f"results/plots/atlas_{i}.png"
            self.assertTrue(os.path.exists(png_path), f"Missing plot artifact: {png_path}")

        # Check report exists and contains every algorithm name
        report_path = "results/reports/failure_atlas.txt"
        self.assertTrue(os.path.exists(report_path), f"Missing report: {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()

        for case in CASES:
            self.assertIn(
                case["algorithm_name"],
                report_text,
                f"Report missing algorithm name: {case['algorithm_name']}"
            )

    def test_determinism(self):
        """
        Asserts running run_case twice returns identical final x (fixed seeds).
        """
        def _get_x(res):
            if hasattr(res, "x"):
                return res.x
            if isinstance(res, dict) and "x" in res:
                return res["x"]
            return None

        for case in CASES:
            case_id = case["id"]
            with self.subTest(case_id=case_id):
                res1 = run_case(case)
                res2 = run_case(case)

                x_good1 = _get_x(res1["good_result"])
                x_good2 = _get_x(res2["good_result"])
                if x_good1 is not None and x_good2 is not None:
                    self.assertEqual(x_good1, x_good2, f"Case {case_id} good_run non-deterministic final x")

                x_bad1 = _get_x(res1["bad_result"])
                x_bad2 = _get_x(res2["bad_result"])
                if x_bad1 is not None and x_bad2 is not None:
                    self.assertEqual(x_bad1, x_bad2, f"Case {case_id} bad_run non-deterministic final x")

    def test_case_07_nesterov_saddle(self):
        """
        Asserts Case 7 good run converges (grad norm < 1e-4, final f close to -1)
        and bad run ends within 1e-3 of [0, 0].
        """
        case_07 = [c for c in CASES if c["id"] == 7][0]
        good_res = case_07["good_run"]()
        bad_res = case_07["bad_run"]()
        self.assertLess(good_res.grad_norm, 1e-4)
        self.assertAlmostEqual(good_res.fun, -1.0, delta=1e-3)
        self.assertLess(abs(bad_res.x[0]), 1e-3)
        self.assertLess(abs(bad_res.x[1]), 1e-3)

    def test_case_descriptions_and_report(self):
        """
        Asserts every case has non-empty good_desc and bad_desc,
        and write_report produces a report containing them.
        """
        for case in CASES:
            self.assertTrue(bool(case.get("good_desc")), f"Case {case['id']} missing good_desc")
            self.assertTrue(bool(case.get("bad_desc")), f"Case {case['id']} missing bad_desc")

        results = [run_case(c) for c in CASES]
        report_path = "results/reports/failure_atlas.txt"
        write_report(results, report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()

        for case in CASES:
            self.assertIn(case["good_desc"], report_text)
            self.assertIn(case["bad_desc"], report_text)

if __name__ == "__main__":
    unittest.main()
