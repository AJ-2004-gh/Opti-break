import unittest
from core.matrices import (
    identity, transpose, mat_vec, mat_mat, outer,
    mat_add, mat_scale, diag, is_symmetric
)

class TestMatrices(unittest.TestCase):
    def test_hand_computed_2x2(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        v = [5.0, 6.0]
        # 1*5 + 2*6 = 17, 3*5 + 4*6 = 39
        self.assertEqual(mat_vec(A, v), [17.0, 39.0])

        B = [[2.0, 0.0], [1.0, 3.0]]
        # [[1*2 + 2*1, 1*0 + 2*3], [3*2 + 4*1, 3*0 + 4*3]] = [[4, 6], [10, 12]]
        self.assertEqual(mat_mat(A, B), [[4.0, 6.0], [10.0, 12.0]])

    def test_hand_computed_3x3(self):
        A = [
            [1.0, 0.0, 2.0],
            [-1.0, 3.0, 1.0],
            [0.0, 2.0, -1.0]
        ]
        v = [1.0, 2.0, 3.0]
        # [1*1 + 0*2 + 2*3, -1*1 + 3*2 + 1*3, 0*1 + 2*2 - 1*3] = [7, 8, 1]
        self.assertEqual(mat_vec(A, v), [7.0, 8.0, 1.0])

        B = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        expected_AB = [
            [0.0, 1.0, 2.0],
            [3.0, -1.0, 1.0],
            [2.0, 0.0, -1.0]
        ]
        self.assertEqual(mat_mat(A, B), expected_AB)

    def test_transpose_2x3(self):
        M = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        expected_T = [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0]
        ]
        self.assertEqual(transpose(M), expected_T)

    def test_additional_matrix_operations(self):
        I3 = identity(3)
        self.assertEqual(I3[0], [1.0, 0.0, 0.0])
        self.assertEqual(I3[1], [0.0, 1.0, 0.0])
        self.assertEqual(I3[2], [0.0, 0.0, 1.0])

        out = outer([1.0, 2.0], [3.0, 4.0])
        self.assertEqual(out, [[3.0, 4.0], [6.0, 8.0]])

        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[0.5, 1.5], [-1.0, 2.0]]
        self.assertEqual(mat_add(A, B), [[1.5, 3.5], [2.0, 6.0]])
        self.assertEqual(mat_scale(2.0, A), [[2.0, 4.0], [6.0, 8.0]])

        D = diag([2.0, 5.0])
        self.assertEqual(D, [[2.0, 0.0], [0.0, 5.0]])

        self.assertTrue(is_symmetric([[1.0, 2.0], [2.0, 3.0]]))
        self.assertFalse(is_symmetric([[1.0, 2.0], [3.0, 4.0]]))

if __name__ == "__main__":
    unittest.main()
