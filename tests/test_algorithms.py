import unittest
from problems.quadratic import make_quadratic, quadratic_f, quadratic_grad, Q_DEFAULT
from problems.rosenbrock import rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess
from problems.nonsmooth import l1_norm, l1_subgradient
from problems.constrained import project_box
from core.matrices import diag, mat_vec
from core.vectors import dot, norm_l2
from algorithms.gradient_descent import (
    gradient_descent_fixed, gradient_descent_backtracking, gradient_descent_exact,
    gradient_descent_armijo
)
from algorithms.momentum import momentum, polyak_momentum
from algorithms.nesterov import nesterov, nesterov_accelerated_gradient
from problems.nonconvex import (
    saddle_f, saddle_grad, saddle_hess, double_well_f, double_well_grad
)
from core.linear_systems import solve, eigenvalues_2x2
from core.matrices import diag, mat_vec, is_symmetric
from core.vectors import dot, norm_l2, vec_sub
from algorithms.newton import (
    newton_method, damped_newton_method, verify_newton_one_step,
    newton_pure, newton_damped
)
from algorithms.adaptive_optimizers import adagrad, rmsprop, adam, adamw
from algorithms.subgradient import (
    subgradient_fixed, subgradient_diminishing, projected_subgradient
)
from algorithms.bfgs import bfgs
from algorithms.regression_optimizers import ista, fista, coordinate_descent

class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        # Strongly convex quadratic with minimum at [1.0, -1.0]
        # f(x) = 0.5 * x^T A x - b^T x where A = [[4, 0], [0, 2]], b = [4, -2]
        self.A = [[4.0, 0.0], [0.0, 2.0]]
        self.b = [4.0, -2.0]
        self.f_quad, self.g_quad, self.h_quad = make_quadratic(self.A, self.b)
        self.x0 = [0.0, 0.0]

    def test_gradient_descent_variants(self):
        res_fixed = gradient_descent_fixed(self.f_quad, self.g_quad, self.x0, alpha=0.1, tol=1e-4)
        self.assertTrue(res_fixed.converged)
        self.assertAlmostEqual(res_fixed.x[0], 1.0, places=2)
        self.assertAlmostEqual(res_fixed.x[1], -1.0, places=2)

        res_bt = gradient_descent_backtracking(self.f_quad, self.g_quad, self.x0, tol=1e-4)
        self.assertTrue(res_bt.converged)
        self.assertAlmostEqual(res_bt.x[0], 1.0, places=2)
        self.assertAlmostEqual(res_bt.x[1], -1.0, places=2)

        res_exact = gradient_descent_exact(self.x0, self.A, self.b, tol=1e-4)
        self.assertTrue(res_exact.converged)
        self.assertAlmostEqual(res_exact.x[0], 1.0, places=2)
        self.assertAlmostEqual(res_exact.x[1], -1.0, places=2)

    def test_gradient_descent_fixed_step_divergence(self):
        # GD fixed with alpha=0.015 on quadratic from [2,1] reaches grad_norm<1e-4;
        # alpha=0.025 ends with message 'diverged' and a much larger f than the start.
        f = quadratic_f
        grad = quadratic_grad
        x0 = [2.0, 1.0]

        res_converge = gradient_descent_fixed(f, grad, x0, alpha=0.015, tol=1e-4, max_iters=2000)
        self.assertTrue(res_converge.converged)
        self.assertLess(res_converge.grad_norm, 1e-4)

        res_diverge = gradient_descent_fixed(f, grad, x0, alpha=0.025, tol=1e-4, max_iters=2000)
        self.assertFalse(res_diverge.converged)
        self.assertEqual(res_diverge.message, "diverged")
        self.assertGreater(res_diverge.fun, res_diverge.f_hist[0])

    def test_gradient_descent_exact_zigzag_and_condition_number(self):
        # Exact-line-search GD: g_k . g_{k+1} is ~0 (|cos| < 1e-6) for the first 10 iterations (zig-zag proof);
        # it converges on kappa=100 but takes many more iterations than on kappa=1.
        Q100 = diag([1.0, 100.0])
        x0 = [10.0, 1.0]
        res100 = gradient_descent_exact(x0, Q100, tol=1e-5, max_iters=1000)
        self.assertTrue(res100.converged)
        self.assertGreaterEqual(res100.iters, 10)

        for k in range(10):
            xk = res100.path[k]
            xk1 = res100.path[k + 1]
            gk = mat_vec(Q100, xk)
            gk1 = mat_vec(Q100, xk1)
            norm_k = norm_l2(gk)
            norm_k1 = norm_l2(gk1)
            self.assertGreater(norm_k, 0.0)
            self.assertGreater(norm_k1, 0.0)
            cos_val = dot(gk, gk1) / (norm_k * norm_k1)
            self.assertLess(abs(cos_val), 1e-6)

        Q1 = diag([1.0, 1.0])
        res1 = gradient_descent_exact(x0, Q1, tol=1e-5, max_iters=1000)
        self.assertTrue(res1.converged)
        self.assertEqual(res1.iters, 1)
        self.assertGreater(res100.iters, res1.iters)

    def test_gradient_descent_armijo_monotone_convergence(self):
        # GD+Armijo converges on the quadratic and its f_hist is monotonically non-increasing.
        f = quadratic_f
        grad = quadratic_grad
        x0 = [2.0, 1.0]

        res = gradient_descent_armijo(f, grad, x0, alpha_init=1.0, rho=0.5, c1=1e-4, tol=1e-5, max_iters=1000)
        self.assertTrue(res.converged)
        self.assertLess(res.grad_norm, 1e-5)
        self.assertAlmostEqual(res.x[0], 0.0, places=3)
        self.assertAlmostEqual(res.x[1], 0.0, places=3)

        self.assertGreater(len(res.f_hist), 1)
        for i in range(len(res.f_hist) - 1):
            self.assertGreaterEqual(res.f_hist[i], res.f_hist[i + 1])

    def test_momentum_and_nesterov(self):
        res_polyak = polyak_momentum(self.f_quad, self.g_quad, self.x0, alpha=0.05, beta=0.8, tol=1e-4)
        self.assertTrue(res_polyak.converged)
        self.assertAlmostEqual(res_polyak.x[0], 1.0, places=2)

        res_nag = nesterov_accelerated_gradient(self.f_quad, self.g_quad, self.x0, alpha=0.05, beta=0.8, tol=1e-4)
        self.assertTrue(res_nag.converged)
        self.assertAlmostEqual(res_nag.x[0], 1.0, places=2)

    def test_momentum_vs_plain_gd(self):
        # Momentum with alpha=0.01, beta=0.9 on the quadratic reaches grad_norm<1e-4
        # in fewer iterations than plain GD with alpha=0.01.
        f = quadratic_f
        grad = quadratic_grad
        x0 = [2.0, 1.0]

        res_gd = gradient_descent_fixed(f, grad, x0, alpha=0.01, tol=1e-4, max_iters=2000)
        res_mom = momentum(f, grad, x0, alpha=0.01, beta=0.9, tol=1e-4, max_iters=2000)

        self.assertTrue(res_gd.converged)
        self.assertTrue(res_mom.converged)
        self.assertLess(res_mom.grad_norm, 1e-4)
        self.assertLess(res_mom.iters, res_gd.iters)

    def test_nesterov_convergence(self):
        # Nesterov converges on the quadratic and on Rosenbrock with a small step (assert f decreases by at least 99% from start).
        # On quadratic
        res_quad = nesterov(quadratic_f, quadratic_grad, [2.0, 1.0], alpha=0.01, beta=0.9, tol=1e-4, max_iters=1000)
        self.assertTrue(res_quad.converged)
        self.assertLess(res_quad.grad_norm, 1e-4)

        # On Rosenbrock
        x0_rosen = [-1.2, 1.0]
        f_start = rosenbrock_2d(x0_rosen)
        res_rosen = nesterov(rosenbrock_2d, rosenbrock_2d_grad, x0_rosen, alpha=0.001, beta=0.9, tol=1e-5, max_iters=2000)
        f_end = res_rosen.fun
        f_decrease = (f_start - f_end) / f_start
        self.assertGreaterEqual(f_decrease, 0.99)

    def test_momentum_and_nesterov_reduce_to_plain_gd(self):
        # With beta=0 both reduce exactly to plain GD (same path as Algorithm 1 within 1e-12).
        f = quadratic_f
        grad = quadratic_grad
        x0 = [2.0, 1.0]
        alpha = 0.015

        res_gd = gradient_descent_fixed(f, grad, x0, alpha=alpha, tol=1e-4, max_iters=50)
        res_mom = momentum(f, grad, x0, alpha=alpha, beta=0.0, tol=1e-4, max_iters=50)
        res_nag = nesterov(f, grad, x0, alpha=alpha, beta=0.0, tol=1e-4, max_iters=50)

        self.assertEqual(len(res_mom.path), len(res_gd.path))
        self.assertEqual(len(res_nag.path), len(res_gd.path))

        for k in range(len(res_gd.path)):
            for j in range(len(x0)):
                self.assertAlmostEqual(res_mom.path[k][j], res_gd.path[k][j], delta=1e-12)
                self.assertAlmostEqual(res_nag.path[k][j], res_gd.path[k][j], delta=1e-12)

    def test_momentum_atlas_failure_checks(self):
        # Failure checks used later by the atlas:
        # 1. momentum with beta=0.99 has a non-monotone f_hist (at least 5 increases)
        f = quadratic_f
        grad = quadratic_grad
        x0 = [2.0, 1.0]

        res_high_mom = momentum(f, grad, x0, alpha=0.01, beta=0.99, tol=1e-4, max_iters=500)
        increases = sum(1 for i in range(len(res_high_mom.f_hist) - 1) if res_high_mom.f_hist[i + 1] > res_high_mom.f_hist[i])
        self.assertGreaterEqual(increases, 5)

        # 2. momentum started at [1,0] on f=x^2-y^2 stays on y=0 and converges to the saddle at the origin
        res_saddle = momentum(saddle_f, saddle_grad, [1.0, 0.0], alpha=0.05, beta=0.9, tol=1e-4, max_iters=500)
        self.assertTrue(res_saddle.converged)
        self.assertAlmostEqual(res_saddle.x[0], 0.0, places=3)
        self.assertAlmostEqual(res_saddle.x[1], 0.0, places=10)
        for pt in res_saddle.path:
            self.assertAlmostEqual(pt[1], 0.0, delta=1e-12)

    def test_adaptive_optimizers(self):
        res_adagrad = adagrad(self.f_quad, self.g_quad, self.x0, lr=0.5, tol=1e-3, max_iters=2000)
        self.assertAlmostEqual(res_adagrad.x[0], 1.0, places=1)

        res_rmsprop = rmsprop(self.f_quad, self.g_quad, self.x0, lr=0.1, tol=1e-3, max_iters=1000)
        self.assertAlmostEqual(res_rmsprop.x[0], 1.0, places=1)

        res_adam = adam(self.f_quad, self.g_quad, self.x0, lr=0.1, tol=1e-3, max_iters=1000)
        self.assertAlmostEqual(res_adam.x[0], 1.0, places=1)

        res_adamw = adamw(self.f_quad, self.g_quad, self.x0, lr=0.1, weight_decay=0.001, tol=1e-3, max_iters=1000)
        self.assertAlmostEqual(res_adamw.x[0], 1.0, places=1)

    def test_newton(self):
        # Pure Newton converges in 1 step on quadratic!
        res_newton = newton_pure(self.f_quad, self.g_quad, self.h_quad, self.x0, tol=1e-5)
        self.assertTrue(res_newton.converged)
        self.assertEqual(res_newton.iters, 1)
        self.assertAlmostEqual(res_newton.x[0], 1.0, places=6)
        self.assertAlmostEqual(res_newton.x[1], -1.0, places=6)

        # Damped Newton
        res_damped = newton_damped(self.f_quad, self.g_quad, self.h_quad, self.x0, tol=1e-5)
        self.assertTrue(res_damped.converged)
        self.assertAlmostEqual(res_damped.x[0], 1.0, places=4)

    def test_bfgs(self):
        res_bfgs = bfgs(rosenbrock_2d, rosenbrock_2d_grad, [0.0, 0.0], tol=1e-4, max_iters=200)
        self.assertTrue(res_bfgs.converged)
        self.assertAlmostEqual(res_bfgs.x[0], 1.0, places=3)
        self.assertAlmostEqual(res_bfgs.x[1], 1.0, places=3)

    def test_subgradient_and_projected(self):
        res_dim = subgradient_diminishing(l1_norm, l1_subgradient, [2.0, -3.0], alpha0=0.5, power=0.5, max_iters=200)
        self.assertLess(res_dim.fun, 0.1)

        proj_fn = lambda pt: project_box(pt, [-0.5, -0.5], [0.5, 0.5])
        res_proj_sub = projected_subgradient(l1_norm, l1_subgradient, proj_fn, [2.0, 2.0], max_iters=100)
        self.assertLessEqual(res_proj_sub.x[0], 0.5 + 1e-6)

    def test_ista_and_fista(self):
        # min 0.5 ||x - [2, -2]||^2 + 0.5 ||x||_1
        # Solution: x0 = S_{0.5}(2) = 1.5, x1 = S_{0.5}(-2) = -1.5
        f_smooth = lambda x: 0.5 * (x[0] - 2.0)**2 + 0.5 * (x[1] + 2.0)**2
        g_smooth = lambda x: [x[0] - 2.0, x[1] + 2.0]
        res_ista = ista(f_smooth, g_smooth, [0.0, 0.0], lmbda=0.5, L=1.0, tol=1e-4)
        self.assertTrue(res_ista.converged)
        self.assertAlmostEqual(res_ista.x[0], 1.5, places=3)
        self.assertAlmostEqual(res_ista.x[1], -1.5, places=3)

        res_fista = fista(f_smooth, g_smooth, [0.0, 0.0], lmbda=0.5, L=1.0, tol=1e-4)
        self.assertTrue(res_fista.converged)
        self.assertAlmostEqual(res_fista.x[0], 1.5, places=3)
        self.assertAlmostEqual(res_fista.x[1], -1.5, places=3)

    def test_newton_spd_quadratic_one_step(self):
        # Newton on any 2-D SPD quadratic lands on the minimizer in exactly 1 iteration (error < 1e-9);
        # verify_newton_one_step agrees.
        Q = [[5.0, 2.0], [2.0, 3.0]]
        c = [1.0, -4.0]
        x0 = [10.0, -5.0]

        def f(x):
            Qx = mat_vec(Q, x)
            return 0.5 * dot(x, Qx) - dot(c, x)

        def grad(x):
            Qx = mat_vec(Q, x)
            return [Qx[i] - c[i] for i in range(len(x))]

        def hess(x):
            return Q

        x_star = solve(Q, c)
        res = newton_method(f, grad, hess, x0, tol=1e-5)
        self.assertTrue(res.converged)
        self.assertEqual(res.iters, 1)

        err = norm_l2([res.x[i] - x_star[i] for i in range(len(x_star))])
        self.assertLess(err, 1e-9)

        dist = verify_newton_one_step(Q, c, x0)
        self.assertLess(dist, 1e-9)
        self.assertAlmostEqual(err, dist, places=12)

    def test_newton_and_damped_rosenbrock(self):
        # Newton converges on Rosenbrock from [-1.2,1] and damped Newton does too with monotone f_hist.
        x0 = [-1.2, 1.0]

        res_newton = newton_method(rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess, x0, tol=1e-5)
        self.assertTrue(res_newton.converged)
        self.assertLess(res_newton.grad_norm, 1e-5)
        self.assertAlmostEqual(res_newton.x[0], 1.0, places=3)
        self.assertAlmostEqual(res_newton.x[1], 1.0, places=3)

        res_damped = damped_newton_method(rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess, x0, tol=1e-5)
        self.assertTrue(res_damped.converged)
        self.assertLess(res_damped.grad_norm, 1e-5)
        self.assertAlmostEqual(res_damped.x[0], 1.0, places=3)
        self.assertAlmostEqual(res_damped.x[1], 1.0, places=3)

        # Monotone f_hist
        self.assertGreater(len(res_damped.f_hist), 1)
        for i in range(len(res_damped.f_hist) - 1):
            self.assertLessEqual(res_damped.f_hist[i + 1], res_damped.f_hist[i] + 1e-12)

    def test_bfgs_properties_and_comparison(self):
        # BFGS on the quadratic and on Rosenbrock reaches grad_norm < 1e-5;
        # BFGS iterations < gradient descent iterations on Rosenbrock;
        # H stays symmetric (check within 1e-10).
        # On quadratic
        res_quad = bfgs(self.f_quad, self.g_quad, self.x0, tol=1e-5)
        self.assertTrue(res_quad.converged)
        self.assertLess(res_quad.grad_norm, 1e-5)
        self.assertTrue(is_symmetric(res_quad.extra["H"], tol=1e-10))

        # On Rosenbrock from [-1.2, 1.0]
        x0 = [-1.2, 1.0]
        res_rosen = bfgs(rosenbrock_2d, rosenbrock_2d_grad, x0, tol=1e-5, max_iters=200)
        self.assertTrue(res_rosen.converged)
        self.assertLess(res_rosen.grad_norm, 1e-5)
        self.assertTrue(is_symmetric(res_rosen.extra["H"], tol=1e-10))

        # Compare with GD
        res_gd = gradient_descent_armijo(rosenbrock_2d, rosenbrock_2d_grad, x0, alpha_init=1.0, tol=1e-5, max_iters=200)
        self.assertLess(res_rosen.iters, res_gd.iters)

    def test_newton_and_bfgs_failures(self):
        # Failure checks:
        # 1. Newton on f=x^2-y^2 from [1,1] lands on [0,0] after 1 step,
        #    where hessian eigenvalues are [-2,2] (saddle, f not a minimum).
        res_saddle = newton_method(saddle_f, saddle_grad, saddle_hess, [1.0, 1.0], tol=1e-5)
        self.assertEqual(res_saddle.iters, 1)
        self.assertAlmostEqual(res_saddle.x[0], 0.0, places=9)
        self.assertAlmostEqual(res_saddle.x[1], 0.0, places=9)
        evals = eigenvalues_2x2(saddle_hess(res_saddle.x))
        self.assertAlmostEqual(evals[0], -2.0, places=9)
        self.assertAlmostEqual(evals[1], 2.0, places=9)

        # 2. Damped Newton on the same function sets non_descent or hits the line-search failure.
        res_damped_saddle = damped_newton_method(saddle_f, saddle_grad, saddle_hess, [1.0, 1.0], tol=1e-5)
        self.assertFalse(res_damped_saddle.converged)
        self.assertTrue(
            res_damped_saddle.extra.get("non_descent", False) or
            res_damped_saddle.extra.get("line_search_failed", False)
        )

        # 3. Newton on Rosenbrock is NOT one step (iterations > 3).
        res_rosen = newton_method(rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess, [-1.2, 1.0], tol=1e-5)
        self.assertGreater(res_rosen.iters, 3)

        # 4. BFGS (Armijo-only) on the double-well f=x^4-2x^2+y^2:
        #    hard-code one start point where sy_hist contains a value <= 0 (or skipped > 0), and assert it.
        x0_dw = [-1.7, 0.0]
        res_dw = bfgs(double_well_f, double_well_grad, x0_dw, tol=1e-5, max_iters=50, line_search='armijo')
        has_non_positive_sy = any(val <= 0.0 for val in res_dw.extra.get("sy_hist", []))
        has_skipped = res_dw.extra.get("skipped", 0) > 0
        self.assertTrue(has_non_positive_sy or has_skipped)
        self.assertTrue(has_non_positive_sy)
        self.assertTrue(has_skipped)

if __name__ == "__main__":
    unittest.main()
