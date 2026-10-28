import unittest
from core.convergence import Result, should_stop

class TestConvergence(unittest.TestCase):
    def test_result_class(self):
        res = Result(
            x=[1.0, 2.0],
            fun=0.05,
            grad_norm=1e-6,
            iters=15,
            converged=True,
            message="Converged: gradient norm below tolerance",
            path=[[0.0, 0.0], [1.0, 2.0]],
            f_hist=[10.0, 0.05],
            g_hist=[2.0, 1e-6],
            extra={"learning_rate": 0.01}
        )
        self.assertEqual(res.x, [1.0, 2.0])
        self.assertEqual(res.fun, 0.05)
        self.assertEqual(res.grad_norm, 1e-6)
        self.assertEqual(res.iters, 15)
        self.assertTrue(res.converged)
        self.assertEqual(len(res.path), 2)
        self.assertEqual(len(res.f_hist), 2)
        self.assertEqual(len(res.g_hist), 2)
        self.assertEqual(res.extra["learning_rate"], 0.01)

    def test_should_stop_cases(self):
        # Case 1: Gradient norm below tolerance -> should stop (True)
        self.assertTrue(should_stop(g_norm=1e-6, k=10, tol=1e-5, max_iters=100))

        # Case 2: Iterations reached max_iters -> should stop (True)
        self.assertTrue(should_stop(g_norm=0.5, k=100, tol=1e-5, max_iters=100))

        # Case 3: Neither condition met -> should NOT stop (False)
        self.assertFalse(should_stop(g_norm=0.5, k=10, tol=1e-5, max_iters=100))

if __name__ == "__main__":
    unittest.main()
