import unittest
from core.vectors import (
    validate_vector, vec_add, vec_sub, scalar_mul, dot,
    norm_l1, norm_l2, norm_inf, hadamard, vec_div_elementwise, zeros
)

class TestVectors(unittest.TestCase):
    def test_required_operations(self):
        # Specific user-requested assertions:
        self.assertEqual(dot([1, 2, 3], [4, 5, 6]), 32.0)
        self.assertEqual(norm_l2([3, 4]), 5.0)
        self.assertEqual(norm_l1([1, -2, 3]), 6.0)
        self.assertEqual(norm_inf([1, -7, 3]), 7.0)

    def test_mismatched_lengths_raise_value_error(self):
        u = [1.0, 2.0]
        v = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            dot(u, v)
        with self.assertRaises(ValueError):
            vec_add(u, v)
        with self.assertRaises(ValueError):
            vec_sub(u, v)
        with self.assertRaises(ValueError):
            hadamard(u, v)
        with self.assertRaises(ValueError):
            vec_div_elementwise(u, v)

    def test_additional_vector_functions(self):
        validate_vector([1.0, -2.5, 3])
        with self.assertRaises(ValueError):
            validate_vector([])
        with self.assertRaises(ValueError):
            validate_vector(["a", 1.0])

        z = zeros(4)
        self.assertEqual(z, [0.0, 0.0, 0.0, 0.0])

        # scalar_mul
        self.assertEqual(scalar_mul(2.5, [2.0, -4.0]), [5.0, -10.0])

        # hadamard
        self.assertEqual(hadamard([2.0, 3.0], [4.0, 5.0]), [8.0, 15.0])

        # vec_div_elementwise
        self.assertEqual(vec_div_elementwise([6.0, 8.0], [2.0, 4.0]), [3.0, 2.0])

if __name__ == "__main__":
    unittest.main()
