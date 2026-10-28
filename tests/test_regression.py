import unittest
from problems.linear_regression import (
    generate_linear_regression_data, linear_regression_loss,
    linear_regression_grad, linear_regression_hess, linear_regression_batch_grad,
    generate_data
)
from problems.logistic_regression import (
    generate_logistic_regression_data, logistic_regression_loss,
    logistic_regression_grad, logistic_regression_hess, sigmoid,
    generate_blobs
)
from problems.lasso import (
    generate_sparse_linear_data, lasso_loss, lasso_smooth_grad,
    soft_threshold, lasso_coordinate_descent_step
)
from core.derivatives import check_gradient, check_hessian
from core.vectors import norm_l2
from problems.quadratic import quadratic_f, quadratic_grad
from problems.rosenbrock import rosenbrock_2d, rosenbrock_2d_grad
from algorithms.regression_optimizers import (
    standardize, gd_linear_regression, sgd_linear_regression,
    minibatch_sgd_linear_regression, logistic_regression_gd
)
from algorithms.adaptive_optimizers import adagrad, rmsprop, adam, adamw

class TestRegression(unittest.TestCase):
    def test_linear_regression_derivatives(self):
        X, y, w_true = generate_linear_regression_data(n_samples=25, n_features=3, seed=42)
        w0 = [0.1, -0.2, 0.5]

        def f(w):
            return linear_regression_loss(w, X, y, l2_reg=0.01)

        def grad_f(w):
            return linear_regression_grad(w, X, y, l2_reg=0.01)

        def hess_f(w):
            return linear_regression_hess(w, X, y, l2_reg=0.01)

        _, _, is_grad_ok = check_gradient(f, grad_f, w0, h=1e-5, tol=1e-4)
        self.assertTrue(is_grad_ok)

        _, _, is_hess_ok = check_hessian(f, hess_f, w0, h=1e-4, tol=1e-3)
        self.assertTrue(is_hess_ok)

    def test_linear_regression_batch_grad(self):
        X, y, _ = generate_linear_regression_data(n_samples=10, n_features=2, seed=42)
        w0 = [0.0, 0.0]
        full_grad = linear_regression_grad(w0, X, y, l2_reg=0.0)
        batch_grad = linear_regression_batch_grad(w0, X, y, list(range(10)), l2_reg=0.0)
        for i in range(2):
            self.assertAlmostEqual(full_grad[i], batch_grad[i], places=7)

    def test_logistic_regression_derivatives(self):
        X, y, _ = generate_logistic_regression_data(n_samples=30, n_features=2, seed=42)
        w0 = [0.2, -0.3]

        def f(w):
            return logistic_regression_loss(w, X, y, l2_reg=0.05)

        def grad_f(w):
            return logistic_regression_grad(w, X, y, l2_reg=0.05)

        def hess_f(w):
            return logistic_regression_hess(w, X, y, l2_reg=0.05)

        _, _, is_grad_ok = check_gradient(f, grad_f, w0, h=1e-5, tol=1e-4)
        self.assertTrue(is_grad_ok)

        _, _, is_hess_ok = check_hessian(f, hess_f, w0, h=1e-4, tol=1e-3)
        self.assertTrue(is_hess_ok)

    def test_lasso_soft_threshold(self):
        self.assertEqual(soft_threshold(3.0, 1.0), 2.0)
        self.assertEqual(soft_threshold(-3.0, 1.0), -2.0)
        self.assertEqual(soft_threshold(0.5, 1.0), 0.0)
        self.assertEqual(soft_threshold(-0.8, 1.0), 0.0)

    def test_lasso_coordinate_descent(self):
        X, y, w_true = generate_sparse_linear_data(n_samples=50, n_features=5, n_informative=2, seed=42)
        w = [0.0] * 5
        lmbda = 0.1
        loss_init = lasso_loss(w, X, y, lmbda)
        for _ in range(20):
            w = lasso_coordinate_descent_step(w, X, y, lmbda)
        loss_final = lasso_loss(w, X, y, lmbda)
        self.assertLess(loss_final, loss_init)

    def test_batch_gd_recovers_parameters(self):
        # Batch GD recovers w=[3,2], b=1 within 0.1 on low-noise data.
        X, y = generate_data(n=200, seed=42, noise=0.01)
        res = gd_linear_regression(X, y, lr=0.1, epochs=200)
        w = res.x[:-1]
        b = res.x[-1]
        self.assertAlmostEqual(w[0], 3.0, delta=0.1)
        self.assertAlmostEqual(w[1], 2.0, delta=0.1)
        self.assertAlmostEqual(b, 1.0, delta=0.1)

    def test_sgd_and_minibatch_performance_and_smoothness(self):
        # SGD and mini-batch (batch 16) reach loss within 2x of batch GD's final loss;
        # mini-batch path is smoother (lower variance of step lengths) than single-sample SGD.
        X, y = generate_data(n=200, seed=42, noise=0.01)
        epochs = 150

        res_bgd = gd_linear_regression(X, y, lr=0.02, epochs=epochs)
        res_sgd = sgd_linear_regression(X, y, lr=0.02, epochs=epochs, seed=42)
        res_mb = minibatch_sgd_linear_regression(X, y, lr=0.02, epochs=epochs, batch_size=16, seed=42)

        final_loss_bgd = res_bgd.f_hist[-1]
        final_loss_sgd = res_sgd.f_hist[-1]
        final_loss_mb = res_mb.f_hist[-1]

        self.assertLessEqual(final_loss_sgd, 2.0 * final_loss_bgd)
        self.assertLessEqual(final_loss_mb, 2.0 * final_loss_bgd)

        # Variance of step lengths across epochs
        def step_length_variance(path):
            diffs = []
            for k in range(len(path) - 1):
                delta = [path[k + 1][j] - path[k][j] for j in range(len(path[k]))]
                diffs.append(norm_l2(delta))
            mean_d = sum(diffs) / float(len(diffs))
            return sum((d - mean_d) ** 2 for d in diffs) / float(len(diffs))

        var_sgd = step_length_variance(res_sgd.path)
        var_mb = step_length_variance(res_mb.path)
        self.assertLess(var_mb, var_sgd)

    def test_unscaled_data_divergence_and_standardize_fix(self):
        # Unscaled data (x2 scaled by 1000) with the lr that works on scaled data diverges
        # (message 'diverged' or loss > 1e6); after standardize it converges.
        X_unscaled, y = generate_data(n=200, seed=42, noise=0.01, scale_x2=1000.0)
        lr = 0.05
        epochs = 100

        res_unscaled = gd_linear_regression(X_unscaled, y, lr=lr, epochs=epochs)
        is_diverged = (res_unscaled.message == "diverged") or (res_unscaled.fun > 1e6) or (res_unscaled.f_hist[-1] > 1e6)
        self.assertTrue(is_diverged)

        # After standardize it converges
        X_std = standardize(X_unscaled)
        res_std = gd_linear_regression(X_std, y, lr=lr, epochs=epochs)
        self.assertNotEqual(res_std.message, "diverged")
        self.assertLess(res_std.f_hist[-1], res_std.f_hist[0])
        self.assertLess(res_std.f_hist[-1], 100.0)

    def test_logistic_regression_accuracy_and_loss(self):
        # Logistic regression reaches accuracy >= 0.95 on separated blobs and the loss is lower at the end than at the start.
        X, y = generate_blobs(100, separation=3.0, seed=42)
        res = logistic_regression_gd(X, y, lr=0.5, epochs=200)

        self.assertGreaterEqual(res.extra["accuracy"], 0.95)
        self.assertLess(res.f_hist[-1], res.f_hist[0])
        self.assertIn("decision_boundary", res.extra)

    def test_logistic_regression_separable_weight_growth(self):
        # On perfectly separable data the weight norm after 2000 epochs is > 1.5x the norm after 200 epochs
        # (no finite minimizer) while accuracy is 1.0.
        X, y = generate_blobs(100, separation=3.0, seed=42)
        res_200 = logistic_regression_gd(X, y, lr=0.1, epochs=200)
        res_2000 = logistic_regression_gd(X, y, lr=0.1, epochs=2000)

        norm_200 = norm_l2(res_200.x[:-1])
        norm_2000 = norm_l2(res_2000.x[:-1])

        self.assertEqual(res_200.extra["accuracy"], 1.0)
        self.assertEqual(res_2000.extra["accuracy"], 1.0)
        self.assertGreater(norm_2000, 1.5 * norm_200)

    def test_adaptive_optimizers_quadratic_reduction(self):
        # Each optimizer reduces f by > 95% on the quadratic from [2,1] with a sensible lr.
        x0 = [2.0, 1.0]
        f_start = quadratic_f(x0)

        optimizers_and_lrs = [
            (adagrad, 0.5),
            (rmsprop, 0.05),
            (adam, 0.1)
        ]

        for opt, lr in optimizers_and_lrs:
            res = opt(quadratic_f, quadratic_grad, x0, lr=lr, max_iters=500, tol=1e-5)
            f_end = res.fun
            reduction = (f_start - f_end) / f_start
            self.assertGreater(reduction, 0.95)

    def test_adam_first_step_magnitude(self):
        # Adam first step with b1=0.9, b2=0.999 has magnitude ~lr per coordinate
        # (check within 1e-6 of lr after bias correction).
        x0 = [2.0, 1.0]
        lr = 0.01

        res = adam(quadratic_f, quadratic_grad, x0, lr=lr, b1=0.9, b2=0.999, max_iters=1)
        step_x0 = abs(res.path[1][0] - res.path[0][0])
        step_x1 = abs(res.path[1][1] - res.path[0][1])

        self.assertAlmostEqual(step_x0, lr, delta=1e-6)
        self.assertAlmostEqual(step_x1, lr, delta=1e-6)

    def test_adagrad_step_size_non_increasing(self):
        # AdaGrad: effective step size sequence is non-increasing.
        x0 = [2.0, 1.0]
        res = adagrad(quadratic_f, quadratic_grad, x0, lr=0.1, max_iters=50, tol=1e-8)
        step_hist = res.extra["effective_step_hist"]
        self.assertGreater(len(step_hist), 1)

        for k in range(len(step_hist) - 1):
            for i in range(len(x0)):
                self.assertGreaterEqual(step_hist[k][i], step_hist[k + 1][i] - 1e-15)

    def test_adaptive_optimizers_atlas_failure_checks(self):
        # Failure checks for the atlas:
        # 1. AdaGrad on Rosenbrock from [-1.2,1] with lr=0.1 leaves f far above 0 after 5000 iterations
        # and effective step < 1e-3 at the end.
        x0_rosen = [-1.2, 1.0]
        res_ada = adagrad(rosenbrock_2d, rosenbrock_2d_grad, x0_rosen, lr=0.1, tol=1e-12, max_iters=5000)
        self.assertGreater(res_ada.fun, 0.01)  # leaves f far above 0
        final_eff_step = res_ada.extra["effective_step_size"]
        self.assertLess(final_eff_step[0], 1e-3)
        self.assertLess(res_ada.extra["step_norm"], 1e-3)

        # 2. RMSProp / Adam with a chosen large lr leave a tail where |x_k - x_{k-1}| stays above 1e-3
        # for the last 200 iterations.
        lr_rmsprop_large = 0.02
        lr_adam_large = 0.01
        res_rms = rmsprop(rosenbrock_2d, rosenbrock_2d_grad, x0_rosen, lr=lr_rmsprop_large, tol=1e-12, max_iters=1000)
        res_adam = adam(rosenbrock_2d, rosenbrock_2d_grad, x0_rosen, lr=lr_adam_large, tol=1e-12, max_iters=1000)

        path_rms = res_rms.path
        path_adam = res_adam.path
        self.assertGreaterEqual(len(path_rms), 201)
        self.assertGreaterEqual(len(path_adam), 201)

        tail_steps_rms = [
            norm_l2([path_rms[k][j] - path_rms[k - 1][j] for j in range(len(x0_rosen))])
            for k in range(len(path_rms) - 200, len(path_rms))
        ]
        tail_steps_adam = [
            norm_l2([path_adam[k][j] - path_adam[k - 1][j] for j in range(len(x0_rosen))])
            for k in range(len(path_adam) - 200, len(path_adam))
        ]

        self.assertEqual(len(tail_steps_rms), 200)
        self.assertEqual(len(tail_steps_adam), 200)

        for s in tail_steps_rms:
            self.assertGreater(s, 1e-3)

        for s in tail_steps_adam:
            self.assertGreater(s, 1e-3)

    def test_update_path_entries(self):
        # Assert update_path has epochs * steps_per_epoch entries for SGD and Mini-batch SGD
        X, y = generate_data(n=60, seed=42, noise=0.01)
        epochs = 5
        # SGD: steps_per_epoch = len(y)
        res_sgd = sgd_linear_regression(X, y, lr=0.01, epochs=epochs, seed=42)
        expected_sgd_steps = epochs * len(y)
        self.assertIn("update_path", res_sgd.extra)
        self.assertEqual(len(res_sgd.extra["update_path"]), expected_sgd_steps)

        # Mini-batch SGD: steps_per_epoch = ceil(n / batch_size)
        batch_size = 16
        res_mb = minibatch_sgd_linear_regression(X, y, lr=0.01, epochs=epochs, batch_size=batch_size, seed=42)
        import math
        expected_mb_steps = epochs * math.ceil(len(y) / batch_size)
        self.assertIn("update_path", res_mb.extra)
        self.assertEqual(len(res_mb.extra["update_path"]), expected_mb_steps)

    def test_unbounded_weights_detector(self):
        from atlas.detectors import unbounded_weights
        from atlas.cases import CASES
        case_11 = [c for c in CASES if c["id"] == 11][0]
        good_res = case_11["good_run"]()
        bad_res = case_11["bad_run"]()

        flag_bad, detail_bad = unbounded_weights(bad_res, good_res=good_res)
        self.assertTrue(flag_bad)
        self.assertIn("w_norm_hist", bad_res.extra)
        self.assertEqual(len(bad_res.extra["w_norm_hist"]), 4001)
        w_hist = bad_res.extra["w_norm_hist"]
        self.assertGreater(w_hist[4000], 2.0 * w_hist[400])
        self.assertGreater(w_hist[-1], w_hist[-500])

if __name__ == "__main__":
    unittest.main()
