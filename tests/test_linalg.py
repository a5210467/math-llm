import math
import pytest
from math_llm.linalg import Vector, Matrix


# ======================================================================
# Vector tests
# ======================================================================

class TestVectorInit:
    def test_basic(self) -> None:
        v = Vector([1.0, 2.0, 3.0])
        assert len(v) == 3

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            Vector([])

    def test_indexing(self) -> None:
        v = Vector([4.0, 5.0, 6.0])
        assert v[0] == 4.0
        assert v[2] == 6.0


class TestVectorArithmetic:
    def test_add(self) -> None:
        assert Vector([1, 2]) + Vector([3, 4]) == Vector([4, 6])

    def test_sub(self) -> None:
        assert Vector([5, 3]) - Vector([2, 1]) == Vector([3, 2])

    def test_mul_scalar(self) -> None:
        assert Vector([1, 2, 3]) * 2 == Vector([2, 4, 6])

    def test_rmul_scalar(self) -> None:
        assert 3 * Vector([1, 2]) == Vector([3, 6])

    def test_neg(self) -> None:
        assert -Vector([1, -2]) == Vector([-1, 2])

    def test_add_dim_mismatch(self) -> None:
        with pytest.raises(ValueError):
            Vector([1, 2]) + Vector([1, 2, 3])


class TestVectorOperations:
    def test_dot_product(self) -> None:
        assert math.isclose(Vector([1, 2, 3]).dot(Vector([4, 5, 6])), 32.0)

    def test_dot_orthogonal(self) -> None:
        assert math.isclose(Vector([1, 0]).dot(Vector([0, 1])), 0.0)

    def test_norm(self) -> None:
        assert math.isclose(Vector([3, 4]).norm(), 5.0)

    def test_norm_unit(self) -> None:
        assert math.isclose(Vector([1, 0, 0]).norm(), 1.0)

    def test_scale(self) -> None:
        assert Vector([1, 2]).scale(0.5) == Vector([0.5, 1.0])

    def test_normalize(self) -> None:
        v = Vector([3, 4]).normalize()
        assert math.isclose(v.norm(), 1.0)
        assert math.isclose(v[0], 0.6)
        assert math.isclose(v[1], 0.8)

    def test_normalize_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            Vector([0, 0]).normalize()

    def test_dot_dim_mismatch(self) -> None:
        with pytest.raises(ValueError):
            Vector([1, 2]).dot(Vector([1, 2, 3]))


# ======================================================================
# Matrix tests
# ======================================================================

class TestMatrixInit:
    def test_shape(self) -> None:
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert m.shape == (2, 3)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            Matrix([])

    def test_jagged_raises(self) -> None:
        with pytest.raises(ValueError):
            Matrix([[1, 2], [3]])

    def test_indexing(self) -> None:
        m = Matrix([[1, 2], [3, 4]])
        assert m[0, 1] == 2.0
        assert m[1, 0] == 3.0


class TestMatrixArithmetic:
    def test_add(self) -> None:
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        assert a + b == Matrix([[6, 8], [10, 12]])

    def test_sub(self) -> None:
        a = Matrix([[5, 6], [7, 8]])
        b = Matrix([[1, 2], [3, 4]])
        assert a - b == Matrix([[4, 4], [4, 4]])

    def test_scalar_mul(self) -> None:
        assert Matrix([[1, 2], [3, 4]]) * 2 == Matrix([[2, 4], [6, 8]])

    def test_rscalar_mul(self) -> None:
        assert 3 * Matrix([[1, 0], [0, 1]]) == Matrix([[3, 0], [0, 3]])

    def test_neg(self) -> None:
        assert -Matrix([[1, -2], [3, 0]]) == Matrix([[-1, 2], [-3, 0]])

    def test_add_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            Matrix([[1, 2]]) + Matrix([[1, 2, 3]])


class TestMatrixMultiply:
    def test_matmul_square(self) -> None:
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        assert a * b == Matrix([[19, 22], [43, 50]])

    def test_matmul_rectangular(self) -> None:
        # (2x3) @ (3x2) => (2x2)
        a = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Matrix([[7, 8], [9, 10], [11, 12]])
        assert a * b == Matrix([[58, 64], [139, 154]])

    def test_matmul_identity(self) -> None:
        a = Matrix([[1, 2], [3, 4]])
        i = Matrix.identity(2)
        assert a * i == a

    def test_matmul_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            Matrix([[1, 2]]) * Matrix([[1, 2]])

    def test_matvec(self) -> None:
        m = Matrix([[1, 2], [3, 4]])
        v = Vector([1, 1])
        assert m.matvec(v) == Vector([3, 7])

    def test_matvec_dim_mismatch(self) -> None:
        with pytest.raises(ValueError):
            Matrix([[1, 2]]).matvec(Vector([1, 2, 3]))


class TestMatrixOperations:
    def test_transpose(self) -> None:
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert m.transpose() == Matrix([[1, 4], [2, 5], [3, 6]])

    def test_transpose_square(self) -> None:
        m = Matrix([[1, 2], [3, 4]])
        assert m.transpose() == Matrix([[1, 3], [2, 4]])

    def test_determinant_2x2(self) -> None:
        assert math.isclose(Matrix([[3, 8], [4, 6]]).determinant(), -14.0)

    def test_determinant_3x3(self) -> None:
        m = Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        assert math.isclose(m.determinant(), -306.0)

    def test_determinant_identity(self) -> None:
        assert math.isclose(Matrix.identity(4).determinant(), 1.0)

    def test_determinant_singular(self) -> None:
        assert math.isclose(Matrix([[1, 2], [2, 4]]).determinant(), 0.0)

    def test_determinant_non_square_raises(self) -> None:
        with pytest.raises(ValueError):
            Matrix([[1, 2, 3]]).determinant()

    def test_inverse_2x2(self) -> None:
        m = Matrix([[4, 7], [2, 6]])
        inv = m.inverse()
        product = m * inv
        assert product == Matrix.identity(2)

    def test_inverse_3x3(self) -> None:
        m = Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
        inv = m.inverse()
        assert m * inv == Matrix.identity(3)

    def test_inverse_singular_raises(self) -> None:
        with pytest.raises((ZeroDivisionError, ValueError)):
            Matrix([[1, 2], [2, 4]]).inverse()

    def test_row_reduce_full_rank(self) -> None:
        # det = 1*(2-0) - 2*(0-1) + 0 = 2+2 = 4, full rank
        m = Matrix([[1, 2, 0], [0, 1, 1], [1, 0, 2]])
        rref = m.row_reduce()
        assert rref == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_row_reduce_rank_deficient(self) -> None:
        m = Matrix([[1, 2, 3], [2, 4, 6]])
        rref = m.row_reduce()
        assert rref == Matrix([[1, 2, 3], [0, 0, 0]])


class TestMatrixClassMethods:
    def test_identity(self) -> None:
        i = Matrix.identity(3)
        assert i.shape == (3, 3)
        assert i[0, 0] == 1.0
        assert i[0, 1] == 0.0

    def test_zeros(self) -> None:
        z = Matrix.zeros(2, 3)
        assert z.shape == (2, 3)
        assert z[0, 0] == 0.0
