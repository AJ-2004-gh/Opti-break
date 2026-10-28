import math
import unittest
from problems.constrained import (
    project_box, project_ball, project_hyperplane,
    quadratic_penalty_objective, quadratic_penalty_gradient,
    log_barrier_objective, log_barrier_gradient, max_constraint_violation,
    equality_qp, redundant_constraints_qp, indefinite_qp, inequality_problem
)
from problems.nonsmooth import nonsmooth_f, nonsmooth_subgradient
from problems.lasso import generate_lasso_data_2d
from algorithms.subgradient import subgradient_method, subgradient_lasso
from algorithms.constrained_optimization import (
    lagrange_solver, second_order_check, kkt_check, licq_check
)
from core.vectors import vec_norm, norm_l2
from core.derivatives import check_gradient

class TestConstraints(unittest.TestCase):
    def test_project_box(self):
        x = [-2.0, 0.5, 3.0]
        lower = [-1.0, 0.0, 1.0]
        upper = [1.0, 1.0, 2.0]
        p = project_box(x, lower, upper)
        self.assertEqual(p, [-1.0, 0.5, 2.0])

    def test_project_ball(self):
        x = [3.0, 4.0]
        p = project_ball(x, radius=2.5)
        self.assertAlmostEqual(vec_norm(p), 2.5)
        self.assertAlmostEqual(p[0], 1.5)
        self.assertAlmostEqual(p[1], 2.0)

    def test_project_hyperplane(self):
        # Hyperplane x0 + x1 = 2
        a = [1.0, 1.0]
        b = 2.0
        x = [0.0, 0.0]
        p = project_hyperplane(x, a, b)
        self.assertAlmostEqual(p[0], 1.0)
        self.assertAlmostEqual(p[1], 1.0)

    def test_quadratic_penalty_derivatives(self):
        # f(x) = x0^2 + x1^2, constraint g(x) = x0 + x1 - 1 <= 0
        def f(x):
            return x[0]**2 + x[1]**2
        def grad_f(x):
            return [2.0 * x[0], 2.0 * x[1]]

        def g(x):
            return x[0] + x[1] - 1.0
        def grad_g(x):
            return [1.0, 1.0]

        rho = 10.0
        pen_f = quadratic_penalty_objective(f, [g], rho)
        pen_grad = quadratic_penalty_gradient(grad_f, [g], [grad_g], rho)

        # Infeasible point x = [1.0, 1.0]
        _, _, is_ok = check_gradient(pen_f, pen_grad, [1.0, 1.0])
        self.assertTrue(is_ok)

        # Feasible point x = [0.2, 0.2]
        _, _, is_ok_feas = check_gradient(pen_f, pen_grad, [0.2, 0.2])
        self.assertTrue(is_ok_feas)

    def test_max_violation(self):
        g1 = lambda x: x[0] - 1.0
        g2 = lambda x: x[1] - 2.0
        self.assertEqual(max_constraint_violation([0.5, 1.0], [g1, g2]), 0.0)
        self.assertEqual(max_constraint_violation([1.5, 1.0], [g1, g2]), 0.5)

    def test_subgradient_diminishing_nonsmooth(self):
        # f=abs(x)+2*abs(y) with diminishing step a=0.5 from [3,2]: best f < 0.05 after 5000 iterations.
        x0 = [3.0, 2.0]
        res = subgradient_method(nonsmooth_f, nonsmooth_subgradient, x0, step_rule=('diminishing', 0.5), max_iters=5000)
        self.assertLess(res.fun, 0.05)

    def test_subgradient_constant_kink_oscillation(self):
        # Same problem with constant step 0.3: the last 100 iterates keep changing sign in x or y
        # and best f stays >= 0.05 (kink oscillation).
        x0 = [3.0, 2.0]
        res = subgradient_method(nonsmooth_f, nonsmooth_subgradient, x0, step_rule=('constant', 0.3), max_iters=1000)
        self.assertGreaterEqual(res.fun, 0.05)

        # The last 100 iterates keep changing sign in x or y
        last_100 = res.path[-100:]
        self.assertEqual(len(last_100), 100)
        sign_changes = 0
        for i in range(len(last_100) - 1):
            if (last_100[i][0] * last_100[i + 1][0] < 0) or (last_100[i][1] * last_100[i + 1][1] < 0):
                sign_changes += 1
        self.assertGreaterEqual(sign_changes, 90)

    def test_subgradient_best_f_hist_monotone(self):
        # Best-so-far f_hist is non-increasing even though raw f is not.
        x0 = [3.0, 2.0]
        res = subgradient_method(nonsmooth_f, nonsmooth_subgradient, x0, step_rule=('constant', 0.3), max_iters=200)

        best_f_hist = res.best_f_hist
        raw_f_hist = res.f_hist

        self.assertGreater(len(best_f_hist), 1)
        for i in range(len(best_f_hist) - 1):
            self.assertGreaterEqual(best_f_hist[i], best_f_hist[i + 1])

        raw_increases = sum(1 for i in range(len(raw_f_hist) - 1) if raw_f_hist[i + 1] > raw_f_hist[i])
        self.assertGreater(raw_increases, 0)

    def test_subgradient_lasso_properties(self):
        # Lasso with strong lam: weights have |w| < 0.05 but at least one is NOT exactly 0.0 (no exact sparsity);
        # with lam=0 it recovers the least squares solution within 0.1.
        X, y = generate_lasso_data_2d(n=50, seed=42, noise=0.01)

        # 1. Strong lambda
        res_strong = subgradient_lasso(X, y, lam=10.0, x0=[1.0, 1.0], step_rule=('diminishing', 0.1), max_iters=2000)
        w_strong = res_strong.x
        self.assertTrue(all(abs(w_i) < 0.05 for w_i in w_strong))
        self.assertTrue(any(w_i != 0.0 for w_i in w_strong))

        # 2. lam=0 recovers least squares solution (~ [2.0, 0.0]) within 0.1
        res_ls = subgradient_lasso(X, y, lam=0.0, x0=[0.0, 0.0], step_rule=('diminishing', 0.2), max_iters=2000)
        w_ls = res_ls.x
        self.assertAlmostEqual(w_ls[0], 2.0, delta=0.1)
        self.assertAlmostEqual(w_ls[1], 0.0, delta=0.1)

    def test_lagrange_solver_equality_qp(self):
        # min x^2 + y^2 s.t. x + y = 1: x = [0.5, 0.5], lambda = -1
        # Sign convention: Lagrangian L(x, lambda) = 0.5 * x^T Q x + c^T x + lambda^T (A x - b).
        # Stationarity: Q x + c + A^T lambda = 0 => [1, 1]^T + lambda * [1, 1]^T = 0 => lambda = -1.
        Q, c, A, b = equality_qp()
        x, lambdas, res = lagrange_solver(Q, c, A, b)
        self.assertAlmostEqual(x[0], 0.5, places=6)
        self.assertAlmostEqual(x[1], 0.5, places=6)
        self.assertAlmostEqual(lambdas[0], -1.0, places=6)
        self.assertLess(res, 1e-10)

        # KKT all true
        # grad f(x) = [2*x0, 2*x1] = [1.0, 1.0]
        grad_f = [2.0 * x[0], 2.0 * x[1]]
        kkt_res = kkt_check(grad_f=grad_f, eq_grads=A, ineq_values=[], ineq_grads=[],
                            x=x, lambdas=lambdas, mus=[], tol=1e-5)
        self.assertTrue(kkt_res["stationarity"])
        self.assertTrue(kkt_res["primal_feasibility"])
        self.assertTrue(kkt_res["dual_feasibility"])
        self.assertTrue(kkt_res["complementary_slackness"])
        self.assertTrue(kkt_res["overall"])

        # Second order check: reduced Hessian is positive definite
        so_res = second_order_check(Q, A, x)
        self.assertTrue(so_res["is_positive_definite"])

    def test_kkt_and_licq_inequality_problem(self):
        # Inequality problem (d): min (x-2)^2 + (y-1)^2 s.t. x^2 + y^2 <= 1.
        # Solution on the circle at (2, 1) / sqrt(5), mu > 0, complementary slackness holds, LICQ holds.
        f, grad_f, g_ineq, grad_g_ineq = inequality_problem()
        sqrt5 = math.sqrt(5.0)
        x_star = [2.0 / sqrt5, 1.0 / sqrt5]
        mu = sqrt5 - 1.0

        self.assertGreater(mu, 0.0)

        # KKT check at x_star
        gf = grad_f(x_star)
        gv = [g_ineq(x_star)]
        gg = [grad_g_ineq(x_star)]
        kkt_res = kkt_check(grad_f=gf, eq_grads=[], ineq_values=gv, ineq_grads=gg,
                            x=x_star, lambdas=[], mus=[mu], tol=1e-5)
        self.assertTrue(kkt_res["stationarity"])
        self.assertTrue(kkt_res["primal_feasibility"])
        self.assertTrue(kkt_res["dual_feasibility"])
        self.assertTrue(kkt_res["complementary_slackness"])
        self.assertTrue(kkt_res["overall"])

        # LICQ check on active constraint gradient
        licq_res = licq_check(gg)
        self.assertTrue(licq_res["holds"])
        self.assertGreater(licq_res["gram_det"], 0.0)
        self.assertAlmostEqual(licq_res["gram_det"], 4.0, places=6)

    def test_constrained_failures(self):
        # Failure (a): redundant constraints raise ValueError
        Q_red, c_red, A_red, b_red = redundant_constraints_qp()
        with self.assertRaises(ValueError):
            lagrange_solver(Q_red, c_red, A_red, b_red)

        # Failure (b): min x^2 - y^2 s.t. x = 1 returns [1, 0],
        # KKT stationarity passes, but second_order_check reports NOT positive definite (constrained maximum along y).
        Q_ind, c_ind, A_ind, b_ind = indefinite_qp()
        x, lambdas, res = lagrange_solver(Q_ind, c_ind, A_ind, b_ind)
        self.assertAlmostEqual(x[0], 1.0, places=6)
        self.assertAlmostEqual(x[1], 0.0, places=6)

        # KKT stationarity passes
        # grad f([1, 0]) = [2*1, -2*0] = [2.0, 0.0]
        grad_f_ind = [2.0 * x[0], -2.0 * x[1]]
        kkt_res = kkt_check(grad_f=grad_f_ind, eq_grads=A_ind, ineq_values=[], ineq_grads=[],
                            x=x, lambdas=lambdas, mus=[], tol=1e-5)
        self.assertTrue(kkt_res["stationarity"])

        # second_order_check reports NOT positive definite
        so_res = second_order_check(Q_ind, A_ind, x)
        self.assertFalse(so_res["is_positive_definite"])

        # Failure (c) LICQ: at (0, 0) with gradients (-2, 0) and (2, 0),
        # licq_check is False, Gram determinant is 0, and kkt_check for minimizing f=y at (0,0)
        # fails stationarity for every choice of multipliers (assert on a grid of mu1, mu2 in [0, 10]).
        active_grads = [[-2.0, 0.0], [2.0, 0.0]]
        licq_res = licq_check(active_grads)
        self.assertFalse(licq_res["holds"])
        self.assertAlmostEqual(licq_res["gram_det"], 0.0, places=9)

        # Grid search over mu1, mu2 in [0, 10]
        grad_f_y = [0.0, 1.0]
        for mu1_int in range(11):
            for mu2_int in range(11):
                mu1 = float(mu1_int)
                mu2 = float(mu2_int)
                kkt_fail = kkt_check(grad_f=grad_f_y, eq_grads=[], ineq_values=[0.0, 0.0],
                                     ineq_grads=active_grads, x=[0.0, 0.0],
                                     lambdas=[], mus=[mu1, mu2], tol=1e-4)
                self.assertFalse(kkt_fail["stationarity"])
                self.assertFalse(kkt_fail["overall"])

if __name__ == "__main__":
    unittest.main()
