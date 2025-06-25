import numpy as np

from pylucid import MultiSet, RectSet, Set


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
