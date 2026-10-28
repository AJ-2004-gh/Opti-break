import os
import math
import shutil
import struct
import tempfile
import unittest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.contour import (
    contour_with_paths, contour_with_constraints,
    contour_with_line_search, failure_pair,
    plot_2d_contour, plot_optimization_paths,
    plot_armijo_line_search
)
from visualization.model_plots import (
    loss_contour_regression, loss_contour_logistic, loss_contour_lasso,
    loss_contour_regression_pair, loss_contour_logistic_pair,
    plot_convergence_histories, plot_regression_fit, plot_lasso_path
)
from core.convergence import Result
from core.vectors import vec_norm
from problems.quadratic import make_quadratic
from problems.rosenbrock import rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess
from algorithms.line_search import backtracking_line_search
from algorithms.newton import newton_method
from algorithms.adaptive_optimizers import adagrad, rmsprop, adam
from problems.logistic_regression import generate_blobs
from atlas.cases import CASES, _c16_bad
from atlas.runner import run_case

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="optibreak_vis_test_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _verify_png_file(self, filepath):
        self.assertTrue(os.path.exists(filepath), f"File {filepath} was not created.")
        file_size = os.path.getsize(filepath)
        self.assertGreater(file_size, 5000, f"File size {file_size} is <= 5 KB.")
        with open(filepath, "rb") as f:
            header = f.read(8)
        self.assertEqual(header, PNG_SIGNATURE, "Invalid 8-byte PNG signature.")
        self.assertEqual(matplotlib.get_backend().lower(), "agg", "Matplotlib Agg backend must be active.")

    def test_contour_with_paths_and_divergence(self):
        # 1. Normal path
        def f(pt):
            return pt[0] ** 2 + pt[1] ** 2

        out_normal = os.path.join(self.temp_dir, "paths_normal.png")
        paths_normal = {
            "GD": [[1.5, 1.5], [1.0, 1.0], [0.5, 0.5], [0.0, 0.0]],
            "Momentum": [[1.5, 1.5], [0.8, 1.2], [0.2, 0.4], [0.0, 0.0]]
        }
        contour_with_paths(f, (-2.0, 2.0), (-2.0, 2.0), paths_normal, out_normal,
                           title="Normal Convergence Paths", optimum=[0.0, 0.0])
        self._verify_png_file(out_normal)

        # 2. Diverging path: must not raise and must produce valid PNG > 5 KB with diverged arrow
        out_diverge = os.path.join(self.temp_dir, "paths_diverging.png")
        paths_diverge = {
            "Stable": [[1.0, 1.0], [0.5, 0.5], [0.1, 0.1]],
            "Diverging": [[1.0, 1.0], [2.5, 3.0], [15.0, 30.0], [1e5, 1e6]]
        }
        try:
            contour_with_paths(f, (-2.0, 2.0), (-2.0, 2.0), paths_diverge, out_diverge,
                               title="Diverging Path Test")
        except Exception as e:
            self.fail(f"Calling contour_with_paths with a diverging path raised an unexpected exception: {e}")

        self._verify_png_file(out_diverge)

    def test_contour_with_constraints(self):
        def f(pt):
            return (pt[0] - 2.0) ** 2 + (pt[1] - 1.0) ** 2

        # Equality: x0 + x1 - 1 = 0
        eq_c = [lambda pt: pt[0] + pt[1] - 1.0]
        # Inequality: x0^2 + x1^2 - 1 <= 0
        ineq_c = [lambda pt: pt[0] ** 2 + pt[1] ** 2 - 1.0]
        # Solution point on boundary
        sol = [2.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0)]
        active_grads = [(sol, [2.0 * sol[0], 2.0 * sol[1]], "grad g")]

        out_file = os.path.join(self.temp_dir, "constraints_test.png")
        contour_with_constraints(f, (-2.0, 3.0), (-2.0, 3.0), eq_c, ineq_c,
                                 {"Solution": sol}, out_file, title="Constrained Contour Test",
                                 active_grad_arrows=active_grads)
        self._verify_png_file(out_file)

    def test_contour_with_line_search(self):
        """Tests that backtracking line search contour includes all trials within axis limits."""
        x0_rb = [-1.2, 1.0]
        g_rb = rosenbrock_2d_grad(x0_rb)
        d_rb = [-g_rb[0], -g_rb[1]]
        ls = backtracking_line_search(rosenbrock_2d, x0_rb, d_rb, g_rb, alpha_init=0.02, rho=0.5, c1=1e-4)
        trial_alphas = ls["trials"]
        accepted_alpha = ls["alpha"]

        self.assertGreaterEqual(len(trial_alphas), 4)

        out_file = os.path.join(self.temp_dir, "line_search_test.png")
        xr, yr = contour_with_line_search(rosenbrock_2d, x0_rb, d_rb, trial_alphas, accepted_alpha, out_file,
                                          title="Line Search Armijo Trials")
        self._verify_png_file(out_file)

        all_alphas = list(trial_alphas)
        if accepted_alpha is not None and accepted_alpha not in all_alphas:
            all_alphas.append(accepted_alpha)
        all_pts = [x0_rb] + [[x0_rb[0] + a * d_rb[0], x0_rb[1] + a * d_rb[1]] for a in all_alphas]
        for p in all_pts:
            self.assertGreaterEqual(p[0], xr[0])
            self.assertLessEqual(p[0], xr[1])
            self.assertGreaterEqual(p[1], yr[0])
            self.assertLessEqual(p[1], yr[1])

    def test_contour_with_constraints_licq(self):
        """Tests LICQ degenerate contour generation and file size > 20 KB."""
        f_licq = lambda pt: pt[1]
        ineq_licq = [
            lambda pt: (pt[0] - 1.0)**2 + pt[1]**2 - 1.0,
            lambda pt: (pt[0] + 1.0)**2 + pt[1]**2 - 1.0
        ]
        arrows_licq = [
            ([0.0, 0.0], [-2.0, 0.0], "grad g1 = [-2, 0]"),
            ([0.0, 0.0], [2.0, 0.0], "grad g2 = [2, 0]")
        ]
        out_file = os.path.join(self.temp_dir, "licq_test.png")
        contour_with_constraints(f_licq, (-3.0, 3.5), (-2.0, 2.0), [], ineq_licq, {"(0,0)": [0.0, 0.0]},
                                 out_file, "Alg 23: Degenerate Problem (rank = 1 < 2: LICQ fails)",
                                 active_grad_arrows=arrows_licq)
        self._verify_png_file(out_file)
        self.assertGreater(os.path.getsize(out_file), 20 * 1024, "Saved LICQ PNG must be larger than 20 KB.")

    def test_case_11_3panel_aspect_ratio(self):
        """Tests that Case 11 3-panel plot has non-empty axes for all 3 panels and width > 2*height."""
        X_ov, y_ov = generate_blobs(40, separation=0.5, seed=42)
        X_sep, y_sep = generate_blobs(40, separation=2.0, seed=42)
        out_file = os.path.join(self.temp_dir, "case11_test.png")
        l_data = {
            "X": X_ov, "y": y_ov,
            "paths": {"GD": [[0.0, 0.0], [0.5, 0.5]]},
            "title": "Overlapping",
            "w_norm_hist": [0.0, 0.707]
        }
        r_data = {
            "X": X_sep, "y": y_sep,
            "paths": {"GD": [[0.0, 0.0], [1.0, 2.0]]},
            "title": "Separable",
            "w_norm_hist": [0.0, 2.236]
        }
        axes = loss_contour_logistic_pair(l_data, r_data, "Weights diverge to infinity on separable data", out_file)
        self.assertIsNotNone(axes)
        self.assertEqual(len(axes), 3)
        ax1, ax2, ax3 = axes

        self.assertGreater(len(ax1.collections), 0, "ax1 missing contour collections")
        self.assertGreater(len(ax1.lines), 0, "ax1 missing line trajectory")
        self.assertGreater(len(ax2.collections), 0, "ax2 missing contour collections (empty middle panel bug!)")
        self.assertGreater(len(ax2.lines), 0, "ax2 missing line trajectory")
        self.assertGreaterEqual(len(ax3.lines), 2, "ax3 must have at least 2 lines (bounded and growing)")

        self._verify_png_file(out_file)
        self.assertGreater(os.path.getsize(out_file), 20 * 1024, "Saved Case 11 PNG must be > 20 KB")
        with open(out_file, "rb") as f:
            data = f.read(24)
        w, h = struct.unpack(">II", data[16:24])
        self.assertGreater(w, 2 * h, f"Case 11 PNG width ({w}) must be > 2 * height ({h}).")
        self.assertGreaterEqual(w, 1200)
        plt.close("all")

    def test_all_atlas_pair_panels_nonempty(self):
        """Renders every atlas pair case and asserts that no data axes is empty."""
        from atlas.cases import CASES
        from atlas.runner import run_case
        for case in CASES:
            res = run_case(case, plot_dir=self.temp_dir, history_dir=self.temp_dir)
            axes = res.get("axes")
            self.assertIsNotNone(axes, f"Case {case['id']} did not return axes")
            for idx, ax in enumerate(axes):
                has_content = (
                    len(ax.collections) > 0 or
                    len(ax.lines) > 0 or
                    len(ax.images) > 0 or
                    len(getattr(ax, "patches", [])) > 0
                )
                self.assertTrue(has_content, f"Case {case['id']} panel {idx} is empty!")
        plt.close("all")

    def test_sample_plots_min_width_1200(self):
        """Tests that multiple sample plots at 150 DPI have width >= 1200 px."""
        f = lambda pt: pt[0] ** 2 + pt[1] ** 2
        out1 = os.path.join(self.temp_dir, "sample_single.png")
        contour_with_paths(f, (-2, 2), (-2, 2), {"Path": [[1.0, 1.0], [0.0, 0.0]]}, out1, title="Single Path")

        out2 = os.path.join(self.temp_dir, "sample_pair.png")
        l_args = {"f": f, "xrange": (-2, 2), "yrange": (-2, 2), "paths": {"L": [[1.0, 1.0], [0.0, 0.0]]}, "title": "L"}
        r_args = {"f": f, "xrange": (-2, 2), "yrange": (-2, 2), "paths": {"R": [[1.0, 1.0], [0.0, 0.0]]}, "title": "R"}
        failure_pair(l_args, r_args, "Sample Verdict", out2)

        out3 = os.path.join(self.temp_dir, "sample_constraints.png")
        contour_with_constraints(f, (-3.0, 3.5), (-2.0, 2.0), [], [lambda pt: pt[0]**2 + pt[1]**2 - 1.0],
                                 {"(0,0)": [0.0, 0.0]}, out3, title="Sample Constraints",
                                 active_grad_arrows=[([0.0, 0.0], [2.0, 0.0], "grad g")])

        for path in [out1, out2, out3]:
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as f:
                data = f.read(24)
            w, _ = struct.unpack(">II", data[16:24])
            self.assertGreaterEqual(w, 1200, f"Plot {path} width {w} is less than 1200 pixels.")

    def test_failure_pair(self):
        def f_works(pt):
            return pt[0] ** 2 + pt[1] ** 2

        def f_breaks(pt):
            return pt[0] ** 2 - pt[1] ** 2

        left_args = {
            "f": f_works,
            "xrange": (-2.0, 2.0),
            "yrange": (-2.0, 2.0),
            "paths": {"Convex GD": [[1.5, 1.5], [0.5, 0.5], [0.0, 0.0]]},
            "title": "Convex Quadratic (Works)"
        }
        right_args = {
            "f": f_breaks,
            "xrange": (-2.0, 2.0),
            "yrange": (-2.0, 2.0),
            "paths": {"Saddle Newton": [[1.0, 0.0], [0.0, 0.0]]},
            "title": "Hyperbolic Saddle (Breaks)"
        }
        out_file = os.path.join(self.temp_dir, "failure_pair_test.png")
        failure_pair(left_args, right_args, "Newton step on saddle lands on stationary point without descent", out_file)
        self._verify_png_file(out_file)

    def test_loss_contour_regression(self):
        # Synthetic dataset y = 2*x + 1 + noise
        X = [[-1.0], [0.0], [1.0], [2.0], [3.0]]
        y = [-1.0, 1.0, 3.0, 5.0, 7.0]
        paths = {
            "Batch GD": [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]],
            "SGD": [[0.0, 0.0], [0.8, 0.2], [1.5, 0.8], [2.0, 1.0]],
            "Mini-batch": [[0.0, 0.0], [0.9, 0.4], [1.8, 0.9], [2.0, 1.0]]
        }
        out_file = os.path.join(self.temp_dir, "regression_contour_test.png")
        loss_contour_regression(X, y, paths, out_file, title="Linear Regression MSE Landscape")
        self._verify_png_file(out_file)

    def test_loss_contour_logistic(self):
        # 2D features for two classes
        X = [
            [-1.5, -1.0], [-1.0, -1.5], [-0.5, -0.8],  # Class 0
            [1.0, 1.2], [1.5, 0.8], [2.0, 1.5]          # Class 1
        ]
        y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        paths = {
            "Logistic GD": [[0.0, 0.0], [0.5, 0.5], [1.2, 1.0]]
        }
        out_file = os.path.join(self.temp_dir, "logistic_contour_test.png")
        loss_contour_logistic(X, y, paths, out_file, title="Logistic Regression BCE Landscape", b=0.0)
        self._verify_png_file(out_file)

    def test_loss_contour_lasso(self):
        X = [
            [1.0, 0.5], [0.8, 1.2], [-0.5, -1.0], [1.5, -0.2]
        ]
        y = [2.0, 1.8, -1.2, 2.5]
        paths = {
            "Subgradient": [[1.5, 1.5], [1.0, 0.8], [0.5, 0.1], [0.2, 0.0]]
        }
        out_file = os.path.join(self.temp_dir, "lasso_contour_test.png")
        loss_contour_lasso(X, y, lam=0.5, paths=paths, filepath=out_file, title="Lasso Loss Landscape")
        self._verify_png_file(out_file)

    def test_legacy_functions(self):
        f, _, _ = make_quadratic([[2.0, 0.0], [0.0, 2.0]], [0.0, 0.0])
        out_legacy1 = os.path.join(self.temp_dir, "legacy_contour.png")
        plot_2d_contour(f, output_path=out_legacy1)
        self._verify_png_file(out_legacy1)

        out_legacy2 = os.path.join(self.temp_dir, "legacy_paths.png")
        paths = {"Alg": [[1.0, 1.0], [0.0, 0.0]]}
        plot_optimization_paths(f, paths, output_path=out_legacy2)
        self._verify_png_file(out_legacy2)

        out_conv = os.path.join(self.temp_dir, "legacy_conv.png")
        r1 = Result([0.0], 0.01, 1e-4, 10, True, "done", f_hist=[10.0, 5.0, 1.0, 0.01])
        plot_convergence_histories({"GD": r1}, output_path=out_conv)
        self._verify_png_file(out_conv)

        out_lasso = os.path.join(self.temp_dir, "legacy_lasso.png")
        plot_lasso_path([0.01, 0.1, 1.0], [[1.0, 2.0], [0.5, 1.0], [0.0, 0.0]], output_path=out_lasso)
        self._verify_png_file(out_lasso)

    def test_alg_17_pure_newton_no_diverged_label(self):
        """Verifies pure Newton converges on Rosenbrock and contour shows no false diverged label."""
        r17 = newton_method(rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess, [-1.2, 1.0], max_iters=20)
        self.assertTrue(r17.converged)
        out_file = os.path.join(self.temp_dir, "newton_test_no_diverge.png")
        ax = contour_with_paths(rosenbrock_2d, (-2, 2), (-4.0, 3.5), {"Alg 17 Newton": r17.path},
                                out_file, "Alg 17: Pure Newton", optimum=[1, 1],
                                converged={"Alg 17 Newton": r17.converged})
        self._verify_png_file(out_file)
        texts = [t.get_text().lower() for t in ax.texts]
        self.assertFalse(any("diverged" in t for t in texts))
        plt.close("all")

    def test_adaptive_optimizers_demo_convergence(self):
        """Verifies AdaGrad, RMSProp, and Adam demo configurations reach f < 1e-4 on Q=diag(1, 100)."""
        f_adapt, g_adapt, _ = make_quadratic([[1.0, 0.0], [0.0, 100.0]], [0.0, 0.0])
        x0 = [2.0, 1.0]
        r12 = adagrad(f_adapt, g_adapt, x0, lr=1.0, max_iters=200)
        r13 = rmsprop(f_adapt, g_adapt, x0, lr=0.08, max_iters=200)
        r14 = adam(f_adapt, g_adapt, x0, lr=0.1, max_iters=300)
        self.assertLess(r12.fun, 1e-4)
        self.assertLess(r13.fun, 1e-4)
        self.assertLess(r14.fun, 1e-4)
        self.assertLess(vec_norm(g_adapt(r12.x)), 1e-3)
        self.assertLess(vec_norm(g_adapt(r13.x)), 1e-3)
        self.assertLess(vec_norm(g_adapt(r14.x)), 1e-3)

    def test_alg_04_armijo_figure_distinct_and_sized(self):
        """Verifies Alg 04 Armijo 1D figure exists, differs from Alg 03, and exceeds 20 KB."""
        x0 = [-1.2, 1.0]
        g = rosenbrock_2d_grad(x0)
        nrm = math.hypot(g[0], g[1])
        d = [-g[0] / nrm, -g[1] / nrm]
        ls = backtracking_line_search(rosenbrock_2d, x0, d, g, alpha_init=1.0, rho=0.5, c1=1e-4)
        out3 = os.path.join(self.temp_dir, "alg_03.png")
        contour_with_line_search(rosenbrock_2d, x0, d, ls["trials"], ls["alpha"], out3, "Alg 3")
        out4 = os.path.join(self.temp_dir, "alg_04.png")
        plot_armijo_line_search(rosenbrock_2d, x0, d, g, 1e-4, ls["trials"], ls["alpha"], out4, "Alg 4")
        self.assertTrue(os.path.exists(out4))
        self.assertGreater(os.path.getsize(out4), 20000)
        with open(out3, "rb") as f3, open(out4, "rb") as f4:
            self.assertNotEqual(f3.read(), f4.read())
        plt.close("all")

    def test_case_08_3panel_aspect_ratio(self):
        """Verifies Case 8 produces a 3-panel comparison wider than 2x its height."""
        res = run_case(CASES[7], plot_dir=self.temp_dir, history_dir=self.temp_dir)
        p8 = os.path.join(self.temp_dir, "atlas_8.png")
        self.assertTrue(os.path.exists(p8))
        with open(p8, "rb") as f:
            data = f.read()
        w, h = struct.unpack(">II", data[16:24])
        self.assertGreater(w, 2 * h)
        axes = res.get("axes")
        self.assertIsNotNone(axes)
        self.assertEqual(len(axes), 3)
        plt.close("all")

    def test_case_16_subgradient_lasso_bounds(self):
        """Verifies Case 16 bad run starts at [1.0, -1.0] with 0 < min |w| < 0.05."""
        res_bad = _c16_bad()
        min_w = min(abs(v) for v in res_bad.x)
        self.assertGreater(min_w, 0.0)
        self.assertLess(min_w, 0.05)

    def test_case_18_start_marker_and_annotation(self):
        """Verifies Case 18 right panel renders a start marker and non-descent rejection annotation."""
        res = run_case(CASES[17], plot_dir=self.temp_dir, history_dir=self.temp_dir)
        axes = res.get("axes")
        self.assertIsNotNone(axes)
        ax2 = axes[1]
        texts = [t.get_text() for t in ax2.texts]
        self.assertTrue(any("non-descent direction: step rejected" in t for t in texts))
        plt.close("all")

if __name__ == "__main__":
    unittest.main()
