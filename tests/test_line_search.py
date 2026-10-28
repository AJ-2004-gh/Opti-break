import unittest
from core.matrices import diag
from algorithms.line_search import (
    exact_line_search_quadratic, armijo_condition, backtracking_line_search
)

class TestLineSearch(unittest.TestCase):
    def test_exact_step_quadratic(self):
        # exact step on Q=diag(1,100), g=[1,1] equals 2/101 within 1e-12
        Q = diag([1.0, 100.0])
        g = [1.0, 1.0]
        d = [1.0, 1.0]
        alpha = exact_line_search_quadratic(g, d, Q)
        self.assertAlmostEqual(alpha, 2.0 / 101.0, delta=1e-12)

        # Also verify along d = g
        self.assertAlmostEqual(exact_line_search_quadratic(g, g, Q), 2.0 / 101.0, delta=1e-12)

    def test_armijo_condition(self):
        # f(x) = x0^2 + x1^2, grad = [2*x0, 2*x1]
        f = lambda pt: pt[0] ** 2 + pt[1] ** 2
        x = [2.0, 2.0]
        g = [4.0, 4.0]
        d = [-4.0, -4.0]  # descent direction

        # Armijo returns True for a tiny step on a descent direction and False for a huge step
        self.assertTrue(armijo_condition(f, x, d, g, alpha=1e-5, c1=1e-4))
        self.assertFalse(armijo_condition(f, x, d, g, alpha=100.0, c1=1e-4))

    def test_backtracking_line_search(self):
        f = lambda pt: pt[0] ** 2 + pt[1] ** 2
        x = [2.0, 2.0]
        g = [4.0, 4.0]
        d = [-4.0, -4.0]

        # Backtracking returns dict with alpha, trials, and failed
        res = backtracking_line_search(f, x, d, g, alpha_init=1.0, rho=0.5, c1=1e-4)
        self.assertFalse(res["failed"])
        self.assertLessEqual(res["alpha"], 1.0)
        self.assertTrue(armijo_condition(f, x, d, g, res["alpha"], c1=1e-4))
        self.assertGreater(len(res["trials"]), 0)

        # Ascent direction returns failed=True
        d_ascent = [4.0, 4.0]
        res_ascent = backtracking_line_search(f, x, d_ascent, g, alpha_init=1.0)
        self.assertTrue(res_ascent["failed"])
        self.assertEqual(res_ascent["alpha"], 0.0)

if __name__ == "__main__":
    unittest.main()
