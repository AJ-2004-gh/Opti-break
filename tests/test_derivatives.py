import unittest
from core.derivatives import numerical_gradient, numerical_hessian

class TestDerivatives(unittest.TestCase):
    def test_numerical_gradient_and_hessian(self):
        # f(x, y) = x^2 + 3*y^2
        def f(pt):
            return pt[0] ** 2 + 3.0 * (pt[1] ** 2)

        x0 = [1.0, 2.0]
        # numerical_gradient of x^2+3y^2 at [1,2] is [2,12] within 1e-5
        g = numerical_gradient(f, x0, h=1e-6)
        self.assertAlmostEqual(g[0], 2.0, delta=1e-5)
        self.assertAlmostEqual(g[1], 12.0, delta=1e-5)

        # numerical_hessian gives [[2,0],[0,6]] within 1e-3
        H = numerical_hessian(f, x0, h=1e-4)
        self.assertAlmostEqual(H[0][0], 2.0, delta=1e-3)
        self.assertAlmostEqual(H[0][1], 0.0, delta=1e-3)
        self.assertAlmostEqual(H[1][0], 0.0, delta=1e-3)
        self.assertAlmostEqual(H[1][1], 6.0, delta=1e-3)

if __name__ == "__main__":
    unittest.main()
