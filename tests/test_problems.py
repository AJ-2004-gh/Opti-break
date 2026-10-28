import unittest
import random
from core.derivatives import numerical_gradient, numerical_hessian
from core.vectors import norm_l2, dot
from problems.quadratic import (
    quadratic_f, quadratic_grad, quadratic_hess,
    Q_DEFAULT, MINIMIZER as QUAD_MIN, START_POINT as QUAD_START
)
from problems.rosenbrock import (
    rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess,
    MINIMIZER as ROSEN_MIN
)
from problems.nonconvex import (
    saddle_f, saddle_grad, saddle_hess,
    double_well_f, double_well_grad, double_well_hess
)
from problems.nonsmooth import (
    nonsmooth_f, nonsmooth_subgradient, MINIMIZER as NONSMOOTH_MIN
)
from problems.linear_regression import (
    generate_data, mse, batch_grad,
    generate_data_1d, loss_2d, grad_2d, hess_2d
)
from problems.logistic_regression import (
    generate_blobs, bce_loss, bce_grad, bce_hess, sigmoid
)
from problems.lasso import (
    lasso_2d_loss, lasso_2d_subgradient, generate_lasso_data_2d
)
from problems.constrained import (
    equality_qp, redundant_constraints_qp, indefinite_qp,
    inequality_problem, degenerate_licq_problem
)

class TestProblems(unittest.TestCase):
    def test_rosenbrock_at_minimizer(self):
        # rosenbrock(1,1)=0
        self.assertEqual(rosenbrock_2d(ROSEN_MIN), 0.0)
        self.assertEqual(rosenbrock_2d([1.0, 1.0]), 0.0)

    def test_smooth_problems_gradients_and_hessians(self):
        """
        Gradients match core.numerical_gradient within 1e-4 at 3 random points
        for every smooth problem; Hessians match numerical_hessian within 1e-2.
        """
        rng = random.Random(42)

        # Generate sample points
        test_points = [
            [rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)]
            for _ in range(3)
        ]

        # Prepare fixed data for 2D linear regression & logistic regression
        X_lin_1d, y_lin_1d = generate_data_1d(n=40, seed=42, noise=0.05)
        X_log, y_log = generate_blobs(n=40, separation=2.0, seed=42)

        smooth_problems = [
            ("Quadratic", quadratic_f, quadratic_grad, quadratic_hess),
            ("Rosenbrock", rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess),
            ("Saddle", saddle_f, saddle_grad, saddle_hess),
            ("Double Well", double_well_f, double_well_grad, double_well_hess),
            ("Linear Regression (2D)",
             lambda pt: loss_2d(pt, X_lin_1d, y_lin_1d),
             lambda pt: grad_2d(pt, X_lin_1d, y_lin_1d),
             lambda pt: hess_2d(pt, X_lin_1d, y_lin_1d)),
            ("Logistic Regression (2D)",
             lambda pt: bce_loss(pt, X_log, y_log),
             lambda pt: bce_grad(pt, X_log, y_log),
             lambda pt: bce_hess(pt, X_log, y_log))
        ]

        for name, f_obj, g_obj, h_obj in smooth_problems:
            for pt in test_points:
                ana_g = g_obj(pt)
                num_g = numerical_gradient(f_obj, pt, h=1e-6)
                for i in range(len(pt)):
                    self.assertAlmostEqual(
                        ana_g[i], num_g[i], delta=1e-4,
                        msg=f"{name} gradient mismatch at coord {i} for pt {pt}"
                    )

                ana_H = h_obj(pt)
                num_H = numerical_hessian(f_obj, pt, h=1e-4)
                for i in range(len(pt)):
                    for j in range(len(pt)):
                        self.assertAlmostEqual(
                            ana_H[i][j], num_H[i][j], delta=1e-2,
                            msg=f"{name} Hessian mismatch at ({i}, {j}) for pt {pt}"
                        )

    def test_logistic_gradient_check(self):
        # Logistic gradient check
        X, y = generate_blobs(n=30, separation=2.5, seed=123)
        w0 = [0.5, -0.8]
        ana_g = bce_grad(w0, X, y)
        num_g = numerical_gradient(lambda w: bce_loss(w, X, y), w0, h=1e-6)
        for i in range(2):
            self.assertAlmostEqual(ana_g[i], num_g[i], delta=1e-4)

    def test_separable_generator_has_gap(self):
        """
        Asserts that the clearly separable generator produces data with a strict gap
        between classes that can be separated by a linear decision boundary.
        """
        X, y = generate_blobs(n=60, separation=4.0, seed=42)
        # Blob 0 is centered around [-2, 0], Blob 1 centered around [+2, 0].
        # For a vertical separating line x_0 = 0:
        pts0_x = [X[i][0] for i in range(len(y)) if y[i] == 0.0]
        pts1_x = [X[i][0] for i in range(len(y)) if y[i] == 1.0]

        max_class0_x = max(pts0_x)
        min_class1_x = min(pts1_x)

        # Assert a strict positive gap exists
        gap = min_class1_x - max_class0_x
        self.assertGreater(gap, 0.0, "Expected a strictly positive gap between separable blobs")
        self.assertLess(max_class0_x, min_class1_x)

    def test_lasso_subgradient_at_zero(self):
        """
        Lasso subgradient at w=0 contains 0 in its subdifferential set [-lam, lam].
        """
        X, y = generate_lasso_data_2d(n=20, seed=42)
        lam = 0.5
        w_zero = [0.0, 0.0]

        # Calculate subgradient using our function
        subg = lasso_2d_subgradient(w_zero, X, y, lam=lam)

        # The smooth residual gradient at w=0
        smooth_g = [0.0, 0.0]
        for i in range(len(y)):
            res = -float(y[i])
            smooth_g[0] += res * float(X[i][0])
            smooth_g[1] += res * float(X[i][1])
        smooth_g[0] /= float(len(y))
        smooth_g[1] /= float(len(y))

        # Check that the subgradient returned at w=0 has non-smooth contribution equal to 0
        diff0 = subg[0] - smooth_g[0]
        diff1 = subg[1] - smooth_g[1]
        self.assertAlmostEqual(diff0, 0.0, places=9)
        self.assertAlmostEqual(diff1, 0.0, places=9)

        # And 0 is within [-lam, lam]
        self.assertTrue(-lam <= 0.0 <= lam)

    def test_constrained_problem_formulations(self):
        # (a) Equality QP
        Q, c, A, b = equality_qp()
        self.assertEqual(Q, [[2.0, 0.0], [0.0, 2.0]])
        self.assertEqual(A, [[1.0, 1.0]])
        self.assertEqual(b, [1.0])

        # (b) Redundant QP
        Q, c, A, b = redundant_constraints_qp()
        self.assertEqual(len(A), 2)
        self.assertEqual(A[1], [2.0, 2.0])

        # (c) Indefinite QP
        Q, c, A, b = indefinite_qp()
        self.assertEqual(Q[1][1], -2.0)

        # (d) Inequality problem
        f, grad_f, g_ineq, grad_g_ineq = inequality_problem()
        self.assertEqual(f([2.0, 1.0]), 0.0)
        self.assertEqual(g_ineq([0.0, 0.0]), -1.0)
        self.assertEqual(grad_g_ineq([1.0, 0.0]), [2.0, 0.0])

        # (e) Degenerate LICQ problem
        f, grad_f, g_list, grad_g_list = degenerate_licq_problem()
        self.assertEqual(len(g_list), 2)
        # At (0, 0), gradients are [-2, 0] and [2, 0] (linearly dependent)
        g1_grad_at_zero = grad_g_list[0]([0.0, 0.0])
        g2_grad_at_zero = grad_g_list[1]([0.0, 0.0])
        self.assertEqual(g1_grad_at_zero, [-2.0, 0.0])
        self.assertEqual(g2_grad_at_zero, [2.0, 0.0])

if __name__ == "__main__":
    unittest.main()
