import unittest

class TestSmoke(unittest.TestCase):
    def test_imports(self):
        import core
        import problems
        import algorithms
        import atlas
        import visualization
        import experiments

        self.assertIsNotNone(core)
        self.assertIsNotNone(problems)
        self.assertIsNotNone(algorithms)
        self.assertIsNotNone(atlas)
        self.assertIsNotNone(visualization)
        self.assertIsNotNone(experiments)

if __name__ == "__main__":
    unittest.main()
