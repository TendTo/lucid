import numpy as np

from pylucid import (
    ConstantTruncatedFourierFeatureMap,
    FeatureMap,
    LinearTruncatedFourierFeatureMap,
    LogTruncatedFourierFeatureMap,
    RectSet,
    TruncatedFourierFeatureMap,
)

DIM = 2
NUM_FREQUENCIES = 5


class TestFeatureMap:
    class TestLinearTruncatedFourierFeatureMap:
        def test_init(self):
            feature_map = LinearTruncatedFourierFeatureMap(
                NUM_FREQUENCIES, np.array([1.0, 1.0]), 1.0, RectSet([-1, -1], [1, 1])
            )
            assert feature_map is not None
            assert isinstance(feature_map, LinearTruncatedFourierFeatureMap)
            assert isinstance(feature_map, TruncatedFourierFeatureMap)
            assert isinstance(feature_map, FeatureMap)
            assert feature_map.num_frequencies == NUM_FREQUENCIES
            assert feature_map.dimension == NUM_FREQUENCIES**DIM * 2 - 1
            assert feature_map.omega.shape == (NUM_FREQUENCIES**DIM, DIM)
            assert feature_map.weights.shape == (NUM_FREQUENCIES**DIM * 2 - 1,)

        def test_clone(self):
            feature_map = LinearTruncatedFourierFeatureMap(
                NUM_FREQUENCIES, np.array([1.0, 1.0]), 1.0, RectSet([-1, -1], [1, 1])
            )
            cloned_feature_map = feature_map.clone()
            assert cloned_feature_map is not None
            assert isinstance(cloned_feature_map, LinearTruncatedFourierFeatureMap)
            assert cloned_feature_map.num_frequencies == feature_map.num_frequencies
            assert np.array_equal(cloned_feature_map.omega, feature_map.omega)
            assert np.array_equal(cloned_feature_map.weights, feature_map.weights)

    class TestConstantTruncatedFourierFeatureMap:
        def test_init(self):
            feature_map = ConstantTruncatedFourierFeatureMap(
                NUM_FREQUENCIES, np.array([1.0, 1.0]), 1.0, RectSet([-1, -1], [1, 1])
            )
            assert feature_map is not None
            assert isinstance(feature_map, ConstantTruncatedFourierFeatureMap)
            assert isinstance(feature_map, TruncatedFourierFeatureMap)
            assert isinstance(feature_map, FeatureMap)
            assert feature_map.num_frequencies == NUM_FREQUENCIES
            assert feature_map.dimension == NUM_FREQUENCIES**DIM * 2 - 1
            assert feature_map.omega.shape == (NUM_FREQUENCIES**DIM, DIM)
            assert feature_map.weights.shape == (NUM_FREQUENCIES**DIM * 2 - 1,)

        def test_clone(self):
            feature_map = ConstantTruncatedFourierFeatureMap(
                NUM_FREQUENCIES, np.array([1.0, 1.0]), 1.0, RectSet([-1, -1], [1, 1])
            )
            cloned_feature_map = feature_map.clone()
            assert cloned_feature_map is not None
            assert isinstance(cloned_feature_map, ConstantTruncatedFourierFeatureMap)
            assert cloned_feature_map.num_frequencies == feature_map.num_frequencies
            assert np.array_equal(cloned_feature_map.omega, feature_map.omega)
            assert np.array_equal(cloned_feature_map.weights, feature_map.weights)

    class TestLogTruncatedFourierFeatureMap:
        def test_init(self):
            feature_map = LogTruncatedFourierFeatureMap(
                NUM_FREQUENCIES, np.array([1.0, 1.0]), 1.0, RectSet([-1, -1], [1, 1])
            )
            assert feature_map is not None
            assert isinstance(feature_map, LogTruncatedFourierFeatureMap)
            assert isinstance(feature_map, TruncatedFourierFeatureMap)
            assert isinstance(feature_map, FeatureMap)
            assert feature_map.num_frequencies == NUM_FREQUENCIES
            assert feature_map.dimension == NUM_FREQUENCIES**DIM * 2 - 1
            assert feature_map.omega.shape == (NUM_FREQUENCIES**DIM, DIM)
            assert feature_map.weights.shape == (NUM_FREQUENCIES**DIM * 2 - 1,)

        def test_clone(self):
            feature_map = LogTruncatedFourierFeatureMap(
                NUM_FREQUENCIES, np.array([1.0, 1.0]), 1.0, RectSet([-1, -1], [1, 1])
            )
            cloned_feature_map = feature_map.clone()
            assert cloned_feature_map is not None
            assert isinstance(cloned_feature_map, LogTruncatedFourierFeatureMap)
            assert cloned_feature_map.num_frequencies == feature_map.num_frequencies
            assert np.array_equal(cloned_feature_map.omega, feature_map.omega)
            assert np.array_equal(cloned_feature_map.weights, feature_map.weights)
