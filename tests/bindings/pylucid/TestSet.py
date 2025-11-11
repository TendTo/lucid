import numpy as np

from pylucid import MultiSet, RectSet, Set, SphereSet

def build_lattice(lb: np.ndarray, ub: np.ndarray, per_dim: int, endpoint: bool) -> np.ndarray:
    # per_dim: number of lattice points per dimension (scalar or 2-tuple)
    per_dim = (per_dim,) * len(lb)
    # For periodic lattices (e.g. [0, 2*pi))
    grids = [np.linspace(l, u, n, endpoint=endpoint) for l, u, n in zip(lb, ub, per_dim)]
    mesh = np.meshgrid(*grids, indexing="xy")
    pts = np.vstack([m.ravel() for m in mesh]).T
    return pts


class TestSet:
    class TestRectSet:
        def test_init(self):
            s = RectSet([-1, -4], [2, 3])
            assert s is not None
            assert isinstance(s, Set)
            assert np.allclose(s.lower_bound, [-1, -4])
            assert np.allclose(s.upper_bound, [2, 3])

        def test_contains(self):
            s = RectSet([-1, -2], [1, 3])
            assert [0, 0] in s
            assert [-1, -2] in s
            assert [1, 3] in s
            assert [2, 2] not in s

        def test_lattice(self):
            s = RectSet([-1, -2], [1, 3])
            lattice = s.lattice(points_per_dim=3, endpoint=True)
            expected_points = build_lattice(np.array([-1, -2]), np.array([1, 3]), per_dim=3, endpoint=True)
            assert lattice.shape == expected_points.shape
            assert np.allclose(lattice, expected_points)

        def test_lattice_no_endpoint(self):
            s = RectSet([-1, -2], [1, 3])
            lattice = s.lattice(points_per_dim=3, endpoint=False)
            expected_points = build_lattice(np.array([-1, -2]), np.array([1, 3]), per_dim=3, endpoint=False)
            assert lattice.shape == expected_points.shape
            assert np.allclose(lattice, expected_points)

        def test_print(self):
            s = RectSet([-1, -2], [1, 3])
            assert str(s).startswith("RectSet")

    class TestSphereSet:
        def test_init(self):
            s = SphereSet([1.0, 1.0], 2.0)
            assert s is not None
            assert isinstance(s, Set)
            assert np.allclose(s.center, [1, 1])
            assert s.radius == 2.0

        def test_contains(self):
            s = SphereSet([1.0, 1.0], 2.0)
            assert [0, 0] in s
            assert [1, 1] in s
            assert [3, 1] in s
            assert [1, 3] in s
            assert [np.cos(np.pi / 4) * 2 + 1.0, np.sin(np.pi / 4) * 2 + 1.0] in s

        def test_print(self):
            s = SphereSet([1.0, 1.0], 2.0)
            assert str(s).startswith("SphereSet")

    class TestMultiSet:
        def test_init(self):
            s = MultiSet(RectSet([-1, -4], [2, 3]), RectSet([5, 5], [6, 6]))
            assert s is not None
            assert isinstance(s, Set)

        def test_contains(self):
            s = MultiSet(RectSet([-1, -4], [2, 3]), RectSet([5, 5], [6, 6]))
            assert [0, 0] in s
            assert [-1, -2] in s
            assert [1, 3] in s
            assert [3, 2] not in s
            assert [5, 5] in s
            assert [6, 6] in s
            assert [7, 7] not in s

        def test_print(self):
            s = MultiSet(RectSet([-1, -4], [2, 3]), RectSet([5, 5], [6, 6]))
            assert str(s).startswith("MultiSet(")
