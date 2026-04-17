from __future__ import annotations
import math
from .vector import Vector


class Matrix:
    """Row-major real matrix with standard linear algebra operations."""

    def __init__(self, rows: list[list[float]]) -> None:
        if not rows or not rows[0]:
            raise ValueError("Matrix must have at least one row and column")
        ncols = len(rows[0])
        if any(len(r) != ncols for r in rows):
            raise ValueError("All rows must have the same length")
        self._data = [list(r) for r in rows]
        self._nrows = len(rows)
        self._ncols = ncols

    # ------------------------------------------------------------------ #
    # Properties / dunder
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> tuple[int, int]:
        return (self._nrows, self._ncols)

    def __getitem__(self, index: tuple[int, int]) -> float:
        r, c = index
        return self._data[r][c]

    def __repr__(self) -> str:
        rows = "\n  ".join(str(r) for r in self._data)
        return f"Matrix([\n  {rows}\n])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            return False
        return all(
            math.isclose(self._data[r][c], other._data[r][c])
            for r in range(self._nrows)
            for c in range(self._ncols)
        )

    def __add__(self, other: Matrix) -> Matrix:
        self._check_same_shape(other)
        return Matrix([
            [self._data[r][c] + other._data[r][c] for c in range(self._ncols)]
            for r in range(self._nrows)
        ])

    def __sub__(self, other: Matrix) -> Matrix:
        self._check_same_shape(other)
        return Matrix([
            [self._data[r][c] - other._data[r][c] for c in range(self._ncols)]
            for r in range(self._nrows)
        ])

    def __mul__(self, other: Matrix | float) -> Matrix:
        if isinstance(other, (int, float)):
            return Matrix([
                [self._data[r][c] * other for c in range(self._ncols)]
                for r in range(self._nrows)
            ])
        # Matrix multiplication
        if self._ncols != other._nrows:
            raise ValueError(
                f"Shape mismatch for matmul: {self.shape} x {other.shape}"
            )
        result = [
            [
                sum(self._data[r][k] * other._data[k][c] for k in range(self._ncols))
                for c in range(other._ncols)
            ]
            for r in range(self._nrows)
        ]
        return Matrix(result)

    def __rmul__(self, scalar: float) -> Matrix:
        return self.__mul__(scalar)

    def __neg__(self) -> Matrix:
        return self * (-1.0)

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #

    def transpose(self) -> Matrix:
        return Matrix([
            [self._data[r][c] for r in range(self._nrows)]
            for c in range(self._ncols)
        ])

    def matvec(self, v: Vector) -> Vector:
        """Matrix-vector product Av."""
        if self._ncols != len(v):
            raise ValueError(f"Shape mismatch: matrix cols={self._ncols}, vector dim={len(v)}")
        return Vector([
            sum(self._data[r][c] * v[c] for c in range(self._ncols))
            for r in range(self._nrows)
        ])

    def determinant(self) -> float:
        """Determinant via cofactor expansion (exact for small matrices)."""
        self._check_square()
        return self._det(self._data)

    def inverse(self) -> Matrix:
        """Inverse via Gauss-Jordan elimination."""
        self._check_square()
        if math.isclose(self.determinant(), 0.0):
            raise ValueError("Matrix is singular and has no inverse")
        n = self._nrows
        # Augment [A | I]
        aug = [self._data[r][:] + [1.0 if r == c else 0.0 for c in range(n)]
               for r in range(n)]
        aug = self._gauss_jordan(aug, n)
        return Matrix([row[n:] for row in aug])

    def row_reduce(self) -> Matrix:
        """Reduced row echelon form (RREF)."""
        aug = self._gauss_jordan([r[:] for r in self._data], self._ncols)
        return Matrix(aug)

    def to_list(self) -> list[list[float]]:
        return [r[:] for r in self._data]

    # ------------------------------------------------------------------ #
    # Class methods
    # ------------------------------------------------------------------ #

    @classmethod
    def identity(cls, n: int) -> Matrix:
        return cls([[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)])

    @classmethod
    def zeros(cls, nrows: int, ncols: int) -> Matrix:
        return cls([[0.0] * ncols for _ in range(nrows)])

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _check_square(self) -> None:
        if self._nrows != self._ncols:
            raise ValueError(f"Matrix must be square, got shape {self.shape}")

    def _check_same_shape(self, other: Matrix) -> None:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

    def _det(self, data: list[list[float]]) -> float:
        n = len(data)
        if n == 1:
            return data[0][0]
        if n == 2:
            return data[0][0] * data[1][1] - data[0][1] * data[1][0]
        total = 0.0
        for c in range(n):
            minor = [row[:c] + row[c + 1:] for row in data[1:]]
            total += ((-1) ** c) * data[0][c] * self._det(minor)
        return total

    def _gauss_jordan(self, aug: list[list[float]], ncols: int) -> list[list[float]]:
        """In-place Gauss-Jordan elimination; returns reduced form."""
        nrows = len(aug)
        pivot_row = 0
        for col in range(ncols):
            # Find pivot
            max_row = max(range(pivot_row, nrows), key=lambda r: abs(aug[r][col]))
            if math.isclose(aug[max_row][col], 0.0):
                continue
            aug[pivot_row], aug[max_row] = aug[max_row], aug[pivot_row]
            # Scale pivot row to 1
            pivot = aug[pivot_row][col]
            aug[pivot_row] = [x / pivot for x in aug[pivot_row]]
            # Eliminate column
            for r in range(nrows):
                if r != pivot_row:
                    factor = aug[r][col]
                    aug[r] = [aug[r][k] - factor * aug[pivot_row][k]
                              for k in range(len(aug[r]))]
            pivot_row += 1
            if pivot_row == nrows:
                break
        return aug
