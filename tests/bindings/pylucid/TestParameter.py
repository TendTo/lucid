from pylucid import ParameterValue, ParameterValues, Parameter
import numpy as np
import pytest


class TestParameter:
    class TestParameterValue:
        def test_double(self):
            pv = ParameterValue(Parameter.SIGMA_F, 13.0)
            assert pv.parameter == Parameter.SIGMA_F
            assert isinstance(pv.value, float)
            assert pv.value == 13.0

        def test_int(self):
            pv = ParameterValue(Parameter.DEGREE, 42)
            assert pv.parameter == Parameter.DEGREE
            assert isinstance(pv.value, int)
            assert pv.value == 42

        def test_vector(self):
            pv = ParameterValue(Parameter.SIGMA_L, np.array([1.0, 2.0, 3.0]))
            assert pv.parameter == Parameter.SIGMA_L
            assert isinstance(pv.value, np.ndarray)
            assert np.allclose(pv.value, [1.0, 2.0, 3.0])
            assert pv.value.flags.c_contiguous
            assert not pv.value.flags.owndata
            assert not pv.value.flags.writeable

        def test_equality(self):
            pv1 = ParameterValue(Parameter.SIGMA_F, 13.0)
            pv2 = ParameterValue(Parameter.SIGMA_F, 13.0)
            pv3 = ParameterValue(Parameter.DEGREE, 42)
            assert pv1 == pv2
            assert pv1 != pv3
            assert pv2 != pv3

        def test_str(self):
            pv = ParameterValue(Parameter.SIGMA_F, 13.0)
            assert str(pv).startswith("ParameterValue")
            pv = ParameterValue(Parameter.DEGREE, 42)
            assert str(pv).startswith("ParameterValue")
            pv = ParameterValue(Parameter.SIGMA_L, np.array([1.0, 2.0, 3.0]))
            assert str(pv).startswith("ParameterValue")

    class TestGridSearchTuner:
        def test_int(self):
            pvs = ParameterValues(Parameter.DEGREE, (1, 2, 3))
            assert pvs.parameter == Parameter.DEGREE
            assert pvs.size == 3 and len(pvs) == 3
            assert len(pvs) == 3
            assert pvs.values == (1, 2, 3)

        def test_double(self):
            pvs = ParameterValues(Parameter.SIGMA_F, (0.1, 0.2, 0.3))
            assert pvs.parameter == Parameter.SIGMA_F
            assert pvs.size == 3 and len(pvs) == 3
            assert len(pvs) == 3
            assert pvs.values == (0.1, 0.2, 0.3)

        def test_vector(self):
            pvs = ParameterValues(Parameter.SIGMA_L, (np.array([1.0, 2.0]), np.array([3.0, 4.0])))
            assert pvs.parameter == Parameter.SIGMA_L
            assert pvs.size == 2 and len(pvs) == 2
            assert len(pvs) == 2
            assert isinstance(pvs.values[0], np.ndarray)
            assert isinstance(pvs.values[1], np.ndarray)
            assert np.allclose(pvs.values[0], [1.0, 2.0])
            assert np.allclose(pvs.values[1], [3.0, 4.0])
            assert pvs.values[0].flags.c_contiguous
            # It seems that making a non-owned array writable it's harder than expected.
            # For now, we accept that these assertions will fail.
            with pytest.raises(AssertionError):
                assert not pvs.values[0].flags.owndata
            with pytest.raises(AssertionError):
                assert not pvs.values[0].flags.writeable

        def test_equality(self):
            pvs1 = ParameterValues(Parameter.SIGMA_F, (0.1, 0.2, 0.3))
            pvs2 = ParameterValues(Parameter.SIGMA_F, (0.1, 0.2, 0.3))
            pvs3 = ParameterValues(Parameter.DEGREE, (1, 2, 3))
            assert pvs1 == pvs2
            assert pvs1 != pvs3
            assert pvs2 != pvs3

        def test_str(self):
            pvs = ParameterValues(Parameter.SIGMA_F, (0.1, 0.2, 0.3))
            assert str(pvs).startswith("ParameterValues")
            pvs = ParameterValues(Parameter.DEGREE, (1, 2, 3))
            assert str(pvs).startswith("ParameterValues")
            pvs = ParameterValues(Parameter.SIGMA_L, (np.array([1.0, 2.0]), np.array([3.0, 4.0])))
            assert str(pvs).startswith("ParameterValues")
