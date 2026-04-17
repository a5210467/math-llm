from __future__ import annotations
import math


class Vector:
    """Immutable real-valued vector with standard linear algebra operations."""

    def __init__(self, components: list[float]) -> None:
        if not components:
            raise ValueError("Vector must have at least one component")
        self._data = list(components)

    # ------------------------------------------------------------------ #
    # Dunder / operator overloading
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> float:
        return self._data[index]

    def __repr__(self) -> str:
        return f"Vector({self._data})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(math.isclose(a, b) for a, b in zip(self._data, other._data))

    def __add__(self, other: Vector) -> Vector:
        self._check_same_dim(other)
        return Vector([a + b for a, b in zip(self._data, other._data)])

    def __sub__(self, other: Vector) -> Vector:
        self._check_same_dim(other)
        return Vector([a - b for a, b in zip(self._data, other._data)])

    def __mul__(self, scalar: float) -> Vector:
        """Scalar multiplication: v * 3."""
        return Vector([x * scalar for x in self._data])

    def __rmul__(self, scalar: float) -> Vector:
        """Scalar multiplication: 3 * v."""
        return self.__mul__(scalar)

    def __neg__(self) -> Vector:
        return Vector([-x for x in self._data])

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #

    def dot(self, other: Vector) -> float:
        """Inner product <self, other>."""
        self._check_same_dim(other)
        return sum(a * b for a, b in zip(self._data, other._data))

    def norm(self) -> float:
        """Euclidean (L2) norm."""
        return math.sqrt(self.dot(self))

    def scale(self, scalar: float) -> Vector:
        """Return a new vector scaled by scalar."""
        return self * scalar

    def normalize(self) -> Vector:
        """Return unit vector in the same direction."""
        n = self.norm()
        if math.isclose(n, 0.0):
            raise ValueError("Cannot normalize the zero vector")
        return self * (1.0 / n)

    def to_list(self) -> list[float]:
        return list(self._data)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _check_same_dim(self, other: Vector) -> None:
        if len(self) != len(other):
            raise ValueError(
                f"Dimension mismatch: {len(self)} vs {len(other)}"
            )
