import numpy as np
import pytest

from pylucid import MontecarloSimulation, RectSet, MultiSet, exception, random


def dynamics(x: np.ndarray) -> np.ndarray:
    return x / 2.0


def get_stochastic_dynamics(sigma: float):
    def f(x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, sigma, size=x.shape)
        return x / 2.0 + noise

    return f


class TestMontecarloSimulation:

    def test_deterministic_all_safe(self):
        # 1D bounds and init such that halving state keeps it safe
        X_bounds = RectSet(np.array([-1.0]), np.array([1.0]))
        X_init = RectSet(np.array([-0.5]), np.array([0.5]))
        X_unsafe = MultiSet(RectSet(np.array([-1.0]), np.array([-0.9])), RectSet(np.array([0.9]), np.array([1.0])))

        sim = MontecarloSimulation()
        lb, ub = sim.safety_probability(X_bounds, X_init, X_unsafe, dynamics, 10, 0.9, 1000)

        assert ub >= lb
        assert ub == pytest.approx(1.0, rel=1e-12)
        assert lb > 0

    def test_stochastic_small_noise_all_safe(self):
        random.seed(42)
        X_bounds = RectSet(np.array([-1.0]), np.array([1.0]))
        X_init = RectSet(np.array([-0.5]), np.array([0.5]))
        X_unsafe = MultiSet(RectSet(np.array([-1.0]), np.array([-0.9])), RectSet(np.array([0.9]), np.array([1.0])))

        sim = MontecarloSimulation()
        f = get_stochastic_dynamics(0.1)
        lb, ub = sim.safety_probability(X_bounds, X_init, X_unsafe, f, 10, 0.9, 1000)

        assert ub >= lb
        assert ub == pytest.approx(1.0, rel=1e-12)
        assert lb > 0

    def test_stochastic_larger_noise_some_unsafe(self):
        random.seed(123)
        X_bounds = RectSet(np.array([-1.0]), np.array([1.0]))
        X_init = RectSet(np.array([-0.5]), np.array([0.5]))
        X_unsafe = MultiSet(RectSet(np.array([-1.0]), np.array([-0.9])), RectSet(np.array([0.9]), np.array([1.0])))

        sim = MontecarloSimulation()
        f = get_stochastic_dynamics(0.7)
        lb, ub = sim.safety_probability(X_bounds, X_init, X_unsafe, f, 10, 0.95, 1000)

        assert ub >= lb
        assert ub < 1.0
        assert lb > 0.0

    def test_invalid_confidence_raises(self):
        X_bounds = RectSet(np.array([-1.0]), np.array([1.0]))
        X_init = RectSet(np.array([-0.5]), np.array([0.5]))
        X_unsafe = MultiSet(RectSet(np.array([-1.0]), np.array([-0.9])), RectSet(np.array([0.9]), np.array([1.0])))
        sim = MontecarloSimulation()

        with pytest.raises(exception.LucidInvalidArgumentException):
            sim.safety_probability(X_bounds, X_init, X_unsafe, dynamics, 5, -0.1, 100)

        with pytest.raises(exception.LucidInvalidArgumentException):
            sim.safety_probability(X_bounds, X_init, X_unsafe, dynamics, 5, 1.1, 100)

    def test_invalid_num_samples_raises(self):
        X_bounds = RectSet(np.array([-1.0]), np.array([1.0]))
        X_init = RectSet(np.array([-0.5]), np.array([0.5]))
        X_unsafe = MultiSet(RectSet(np.array([-1.0]), np.array([-0.9])), RectSet(np.array([0.9]), np.array([1.0])))
        sim = MontecarloSimulation()

        with pytest.raises(exception.LucidInvalidArgumentException):
            sim.safety_probability(X_bounds, X_init, X_unsafe, dynamics, 5, 0.9, 0)

    def test_dimension_mismatch_raises(self):
        X_bounds = RectSet(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        X_init = RectSet(np.array([-0.5]), np.array([0.5]))
        X_unsafe = MultiSet(RectSet(np.array([-1.0]), np.array([-0.9])))
        sim = MontecarloSimulation()

        with pytest.raises(exception.LucidInvalidArgumentException):
            sim.safety_probability(X_bounds, X_init, X_unsafe, dynamics, 5, 0.9, 100)
