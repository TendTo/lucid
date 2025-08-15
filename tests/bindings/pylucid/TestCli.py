import json
import os
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pyparsing as pp
import pytest
import yaml

from pylucid import (
    AlglibOptimiser,
    ConstantTruncatedFourierFeatureMap,
    GaussianKernel,
    GurobiOptimiser,
    HighsOptimiser,
    KernelRidgeRegressor,
    LinearTruncatedFourierFeatureMap,
    LogTruncatedFourierFeatureMap,
    MultiSet,
    OptimiserAction,
    RectSet,
    log,
)
from pylucid.cli import (
    ConfigAction,
    Configuration,
    EstimatorAction,
    FeatureMapAction,
    FloatOrNVectorAction,
    KernelAction,
    NMatrixAction,
    SystemDynamicsAction,
    arg_parser,
    type_set,
    type_valid_path,
)


class TestCli:
    class TestArgParser:
        """Test the argument parser functionality"""

        def test_parser_creation(self):
            """Test that we can create the argument parser without errors"""
            parser = arg_parser()
            assert parser is not None
            assert parser.prog == "pylucid"

        def test_default_values(self):
            """Test that default values are set correctly"""
            parser = arg_parser()
            args = parser.parse_args([])

            # Check some default values
            assert args.verbose == log.LOG_INFO
            assert args.seed == -1
            assert args.gamma == 1.0
            assert args.c_coefficient == 1.0
            assert args.lambda_ == 1e-6
            assert args.num_samples == 1000
            assert args.time_horizon == 5
            assert args.noise_scale == 0.01
            assert args.epsilon == 0.0
            assert args.b_kappa == 1.0
            assert args.b_norm == 1.0
            assert not args.plot
            assert not args.verify
            assert args.problem_log_file == ""
            assert args.iis_log_file == ""
            assert args.estimator == KernelRidgeRegressor
            assert args.kernel == GaussianKernel
            assert args.feature_map == LinearTruncatedFourierFeatureMap

        def test_parse_custom_values(self):
            """Test parsing custom command line values"""
            args = arg_parser().parse_args(
                [
                    "--verbose",
                    "5",
                    "--seed",
                    "42",
                    "--gamma",
                    "0.9",
                    "--c_coefficient",
                    "1.2",
                    "--lambda",
                    "0.001",
                    "--num_samples",
                    "500",
                    "--time_horizon",
                    "20",
                    "--noise_scale",
                    "0.05",
                    "--plot",
                    "--verify",
                    "--problem_log_file",
                    "problem.txt",
                    "--iis_log_file",
                    "iis.txt",
                    "--feature_map",
                    "ConstantTruncatedFourierFeatureMap",
                    "--estimator",
                    "KernelRidgeRegressor",
                    "--kernel",
                    "GaussianKernel",
                    "--optimiser",
                    "AlglibOptimiser",
                    "--epsilon",
                    "0.1",
                    "--b_kappa",
                    "0.5",
                    "--b_norm",
                    "2.0",
                ],
            )

            # Check that values were correctly parsed
            assert args.verbose == 5
            assert args.seed == 42
            assert args.gamma == 0.9
            assert args.c_coefficient == 1.2
            assert args.lambda_ == 0.001
            assert args.num_samples == 500
            assert args.time_horizon == 20
            assert args.noise_scale == 0.05
            assert args.plot is True
            assert args.verify is True
            assert args.problem_log_file == "problem.txt"
            assert args.iis_log_file == "iis.txt"
            assert args.feature_map == ConstantTruncatedFourierFeatureMap
            assert args.estimator == KernelRidgeRegressor
            assert args.kernel == GaussianKernel
            assert args.optimiser == AlglibOptimiser
            assert args.epsilon == 0.1
            assert args.b_kappa == 0.5
            assert args.b_norm == 2.0

    class TestTypeConversions:
        """Test type conversion functions used in argument parsing"""

        def test_type_valid_path(self):
            """Test path validation function"""
            # Test with nonexistent file
            with pytest.raises(Exception):
                type_valid_path("nonexistent_file.yaml")

            # Test with valid file types
            for ext in [".py", ".yaml", ".json", ".yml"]:
                with NamedTemporaryFile(suffix=ext) as tmp:
                    path = type_valid_path(tmp.name)
                    assert isinstance(path, Path)
                    assert path.suffix == ext

        def test_type_set(self):
            """Test parsing sets from strings"""
            # Test RectSet parsing
            rect_set = type_set("RectSet([1.0, 2.0], [3.0, 4.0])")
            assert isinstance(rect_set, RectSet)
            assert np.array_equal(rect_set.lower_bound, [1.0, 2.0])
            assert np.array_equal(rect_set.upper_bound, [3.0, 4.0])

            # Test MultiSet parsing
            multi_set = type_set("MultiSet([RectSet([1.0, 2.0], [3.0, 4.0]), RectSet([5.0, 6.0], [7.0, 8.0])])")
            assert isinstance(multi_set, MultiSet)
            assert len(multi_set) == 2

            # Test invalid syntax
            with pytest.raises(pp.ParseException):
                type_set("InvalidSet([1.0, 2.0], [3.0, 4.0])")

        def test_type_estimator(self):
            """Test parsing estimator types from strings"""
            # Test valid estimator string
            args = Configuration()
            action = EstimatorAction(option_strings=None, dest="estimator")
            action(None, args, "KernelRidgeRegressor")
            assert args.estimator == KernelRidgeRegressor

            # Test invalid estimator string
            with pytest.raises(ValueError):
                action(None, args, "InvalidEstimator")

        def test_type_kernel(self):
            """Test parsing kernel types from strings"""
            # Test valid kernel string
            args = Configuration()
            action = KernelAction(option_strings=None, dest="kernel")
            action(None, args, "GaussianKernel")
            assert args.kernel == GaussianKernel

            # Test invalid kernel string
            with pytest.raises(ValueError):
                action(None, args, "InvalidKernel")

        def test_type_feature_map(self):
            """Test parsing feature map types from strings"""
            # Test valid feature map strings
            args = Configuration()
            action = FeatureMapAction(option_strings=None, dest="feature_map")
            action(None, args, "LinearTruncatedFourierFeatureMap")
            assert args.feature_map == LinearTruncatedFourierFeatureMap
            action(None, args, "ConstantTruncatedFourierFeatureMap")
            assert args.feature_map == ConstantTruncatedFourierFeatureMap
            action(None, args, "LogTruncatedFourierFeatureMap")
            assert args.feature_map == LogTruncatedFourierFeatureMap

            # Test invalid feature map string
            with pytest.raises(ValueError):
                action(None, args, "InvalidFeatureMap")

        def test_type_optimiser(self):
            """Test parsing optimiser types from strings"""
            # Test valid optimiser string
            args = Configuration()
            action = OptimiserAction(option_strings=None, dest="optimiser")
            action(None, args, "AlglibOptimiser")
            assert args.optimiser == AlglibOptimiser
            action(None, args, "GurobiOptimiser")
            assert args.optimiser == GurobiOptimiser
            action(None, args, "HighsOptimiser")
            assert args.optimiser == HighsOptimiser

            # Test invalid optimiser string
            with pytest.raises(ValueError):
                action(None, args, "InvalidOptimiser")

        def test_type_float_or_n_vector(self):
            """Test parsing float or vector arguments"""
            # Test single float value
            args = Configuration()
            action = FloatOrNVectorAction(option_strings=None, dest="sigma_l")
            action(None, args, ["1.5"], None)
            assert isinstance(args.sigma_l, float)
            assert args.sigma_l == 1.5

            # Test vector of floats
            action(None, args, ["1.0", "2.0", "3.0"], None)
            assert isinstance(args.sigma_l, np.ndarray)
            assert np.array_equal(args.sigma_l, np.array([1.0, 2.0, 3.0]))

        def test_type_float_or_n_vector_invalid(self):
            """Test invalid float or vector arguments"""
            args = Configuration()
            action = FloatOrNVectorAction(option_strings=None, dest="sigma_l")

            # Test with invalid string
            with pytest.raises(ValueError):
                action(None, args, ["invalid"], None)

            # Test with mixed types
            with pytest.raises(ValueError):
                action(None, args, ["1.0", "2.0", "invalid"], None)

    class TestNMatrixAction:
        """Test the NMatrixAction for parsing matrix data"""

        def test_numpy_array_input(self):
            """Test parsing when input is already a numpy array"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()
            test_array = np.array([[1.0, 2.0], [3.0, 4.0]])

            action(None, namespace, test_array, None)
            assert np.array_equal(namespace.x_samples, test_array)
            assert namespace.x_samples.dtype == np.float64

        def test_json_string_input(self):
            """Test parsing JSON string input"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()
            json_string = "[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]"

            action(None, namespace, json_string, None)
            expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            assert np.array_equal(namespace.x_samples, expected)
            assert namespace.x_samples.dtype == np.float64

        def test_invalid_json_string(self):
            """Test parsing invalid JSON string"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()
            invalid_json = "[[1.0, 2.0], [3.0, 4.0"  # Missing closing bracket

            with pytest.raises(ValueError):
                action(None, namespace, invalid_json, None)

        def test_npy_file_input(self):
            """Test parsing .npy file input"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()
            test_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

            # Create temporary .npy file
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                np.save(tmp.name, test_data)
                tmp_path = tmp.name

            try:
                action(None, namespace, tmp_path, None)
                assert np.array_equal(namespace.x_samples, test_data)
                assert namespace.x_samples.dtype == np.float64
            finally:
                os.unlink(tmp_path)

        def test_npz_file_input(self):
            """Test parsing .npz file input"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()
            test_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

            # Create temporary .npz file
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                np.savez(tmp.name, data=test_data)
                tmp_path = tmp.name

            try:
                action(None, namespace, tmp_path, None)
                # npz files are loaded as dict-like objects, so we need to handle this
                loaded_data = np.load(tmp_path, allow_pickle=True)
                # The action should handle this appropriately
                assert hasattr(namespace, "x_samples")
            finally:
                os.unlink(tmp_path)

        def test_csv_file_input(self):
            """Test parsing .csv file input"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()
            test_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

            # Create temporary .csv file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
                np.savetxt(tmp, test_data, delimiter=",")
                tmp_path = tmp.name

            try:
                action(None, namespace, tmp_path, None)
                assert np.array_equal(namespace.x_samples, test_data)
                assert namespace.x_samples.dtype == np.float64
            finally:
                os.unlink(tmp_path)

        def test_csv_file_with_nan_values(self):
            """Test parsing .csv file with NaN values (should be filtered out)"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()

            # Create temporary .csv file with NaN values
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
                tmp.write("1.0,2.0\n")
                tmp.write("3.0,4.0\n")
                tmp.write("nan,6.0\n")  # This row should be filtered out
                tmp.write("7.0,8.0\n")
                tmp_path = tmp.name

            try:
                action(None, namespace, tmp_path, None)
                expected = np.array([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]])
                assert np.array_equal(namespace.x_samples, expected)
                assert namespace.x_samples.dtype == np.float64
            finally:
                os.unlink(tmp_path)

        def test_empty_csv_file(self):
            """Test parsing empty .csv file"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()

            # Create empty .csv file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                with pytest.raises(Exception):
                    action(None, namespace, tmp_path, None)
            finally:
                os.unlink(tmp_path)

        def test_nonexistent_file(self):
            """Test parsing nonexistent file"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()

            with pytest.raises(Exception):
                action(None, namespace, "nonexistent_file.npy", None)

        def test_unsupported_file_extension(self):
            """Test parsing file with unsupported extension"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()

            # Create temporary file with unsupported extension
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                tmp.write(b"some text data")
                tmp_path = tmp.name

            try:
                # Should fall back to JSON parsing, which should fail
                with pytest.raises(Exception):
                    action(None, namespace, tmp_path, None)
            finally:
                os.unlink(tmp_path)

        def test_1d_array_error(self):
            """Test that 1D arrays are properly handled as errors"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()

            # Test with 1D JSON array
            json_string = "[1.0, 2.0, 3.0, 4.0]"
            with pytest.raises(ValueError):
                action(None, namespace, json_string, None)

        def test_mixed_data_types_in_json(self):
            """Test JSON with mixed data types (should convert to float)"""
            action = NMatrixAction(option_strings=None, dest="x_samples")
            namespace = Configuration()

            # JSON with mixed int and float
            json_string = "[[1, 2.5], [3.0, 4], [5, 6.7]]"
            action(None, namespace, json_string, None)

            expected = np.array([[1.0, 2.5], [3.0, 4.0], [5.0, 6.7]])
            assert np.array_equal(namespace.x_samples, expected)
            assert namespace.x_samples.dtype == np.float64

    class TestSystemDynamicsAction:
        """Test the SystemDynamicsAction for parsing system dynamics"""

        def test_single_dimension(self):
            """Test parsing a single-dimension system dynamics"""
            action = SystemDynamicsAction(option_strings=None, dest="system_dynamics")
            namespace = Configuration()

            action(None, namespace, ["x1 / 2"], None)
            assert namespace.system_dynamics is not None

            # Test the function evaluates correctly
            x_input = np.array([[2.0], [4.0]])
            result = namespace.system_dynamics(x_input)
            assert np.array_equal(result, np.array([[1.0], [2.0]]))

        def test_multi_dimension(self):
            """Test parsing multi-dimensional system dynamics"""
            action = SystemDynamicsAction(option_strings=None, dest="system_dynamics")
            namespace = Configuration()

            action(None, namespace, ["x1 + x2", "x1 - x2"], None)
            assert namespace.system_dynamics is not None

            # Test the function evaluates correctly
            x_input = np.array([[1.0, 2.0], [3.0, 4.0]])
            expected = np.array([[3.0, -1.0], [7.0, -1.0]])
            result = namespace.system_dynamics(x_input)
            assert np.array_equal(result, expected)

        def test_complex_expressions(self):
            """Test parsing complex mathematical expressions"""
            action = SystemDynamicsAction(option_strings=None, dest="system_dynamics")
            namespace = Configuration()

            action(None, namespace, ["sin(x1) + cos(x2)", "exp(x1) * x2"], None)
            assert namespace.system_dynamics is not None

            # Test the function evaluates correctly
            x_input = np.array([[0.0, np.pi / 2], [np.pi / 2, 0.0]])
            expected = np.array(
                [
                    [np.sin(0.0) + np.cos(np.pi / 2), np.exp(0.0) * np.pi / 2],
                    [np.sin(np.pi / 2) + np.cos(0.0), np.exp(np.pi / 2) * 0.0],
                ]
            )
            result = namespace.system_dynamics(x_input)
            assert np.allclose(result, expected)

    class TestFloatOrNVectorAction:
        """Test the FloatOrNVectorAction for parsing float or vector arguments"""

        def test_single_float(self):
            """Test parsing a single float value"""
            action = FloatOrNVectorAction(option_strings=None, dest="sigma_l")
            namespace = Configuration()

            action(None, namespace, [1.5], None)
            assert isinstance(namespace.sigma_l, float)
            assert namespace.sigma_l == 1.5

        def test_vector(self):
            """Test parsing a vector of float values"""
            action = FloatOrNVectorAction(option_strings=None, dest="sigma_l")
            namespace = Configuration()

            action(None, namespace, [1.0, 2.0, 3.0], None)
            assert isinstance(namespace.sigma_l, np.ndarray)
            assert np.array_equal(namespace.sigma_l, np.array([1.0, 2.0, 3.0]))

    class TestConfigAction:
        """Test the ConfigAction for parsing configuration files"""

        @pytest.fixture
        def yaml_config(self) -> "tuple[dict, Path]":
            """Create a temporary YAML config file"""
            with NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
                yaml_dict = {
                    "verbose": 4,
                    "seed": 42,
                    "system_dynamics": ["x1 / 2"],
                    "X_bounds": [{"RectSet": {"lower": [-1], "upper": [1]}}],
                    "X_init": [{"RectSet": {"lower": [-0.5], "upper": [0.5]}}],
                    "X_unsafe": [
                        {"RectSet": {"lower": [-1], "upper": [-0.9]}},
                        {"RectSet": {"lower": [0.9], "upper": [1]}},
                    ],
                    "gamma": 1.0,
                    "c_coefficient": 1.0,
                    "lambda": 0.001,
                    "num_samples": 1000,
                    "time_horizon": 5,
                    "epsilon": 5.0,
                    "b_kappa": 6.0,
                    "b_norm": 7.0,
                    "sigma_f": 15.0,
                    "sigma_l": 1.5,
                    "num_frequencies": 4,
                    "oversample_factor": 2.0,
                    "noise_scale": 0.01,
                    "plot": True,
                    "verify": True,
                    "feature_map": "ConstantTruncatedFourierFeatureMap",
                    "estimator": "KernelRidgeRegressor",
                    "kernel": "GaussianKernel",
                }
                yaml.dump(yaml_dict, tmp)

            yield yaml_dict, Path(tmp.name)
            os.unlink(tmp.name)

        @pytest.fixture
        def json_config_file(self) -> "tuple[dict, Path]":
            """Create a temporary JSON config file"""
            with NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
                json_dict = {
                    "verbose": 1,
                    "seed": 15124,
                    "system_dynamics": ["x1**2", "x1 + x2"],
                    "X_bounds": "RectSet([-2, -2], [2, 2])",
                    "X_init": "RectSet([-1, -1], [1, 1])",
                    "X_unsafe": "RectSet([1.5, 1.5], [2, 2])",
                    "epsilon": 6.0,
                    "b_kappa": 7.0,
                    "b_norm": 8.0,
                    "gamma": 0.12,
                    "c_coefficient": 0.5,
                    "lambda": 0.131,
                    "sigma_f": 10555.0,
                    "sigma_l": [71.0, 2.0],
                    "feature_map": "ConstantTruncatedFourierFeatureMap",
                    "noise_scale": 0.761,
                }
                json.dump(json_dict, tmp)

            yield json_dict, Path(tmp.name)
            os.unlink(tmp.name)

        def test_yaml_config(self, yaml_config: "tuple[dict, Path]"):
            """Test parsing a YAML configuration file"""
            yaml_dict, yaml_path = yaml_config
            action = ConfigAction(option_strings=None, dest="input")
            namespace: Configuration = arg_parser().parse_args([])
            action(None, namespace, yaml_path, None)

            # Check that values were loaded correctly
            assert namespace.verbose == yaml_dict["verbose"]
            assert namespace.seed == yaml_dict["seed"]
            assert namespace.system_dynamics is not None
            assert isinstance(namespace.X_bounds, RectSet)
            assert isinstance(namespace.X_init, RectSet)
            assert isinstance(namespace.X_unsafe, MultiSet)
            assert namespace.gamma == yaml_dict["gamma"]

            assert namespace.c_coefficient == yaml_dict["c_coefficient"]
            assert namespace.lambda_ == yaml_dict["lambda"]
            assert namespace.num_samples == yaml_dict["num_samples"]
            assert namespace.time_horizon == yaml_dict["time_horizon"]
            assert namespace.sigma_f == yaml_dict["sigma_f"]
            assert namespace.sigma_l == yaml_dict["sigma_l"]
            assert namespace.epsilon == yaml_dict["epsilon"]
            assert namespace.b_kappa == yaml_dict["b_kappa"]
            assert namespace.b_norm == yaml_dict["b_norm"]
            assert namespace.feature_map == ConstantTruncatedFourierFeatureMap

            assert namespace.noise_scale == yaml_dict["noise_scale"]
            assert namespace.plot is yaml_dict["plot"]
            assert namespace.verify is yaml_dict["verify"]
            assert namespace.problem_log_file == ""
            assert namespace.iis_log_file == ""
            assert namespace.estimator == KernelRidgeRegressor
            assert namespace.kernel == GaussianKernel

        def test_json_config(self, json_config_file: "tuple[dict, Path]"):
            """Test parsing a JSON configuration file"""
            json_dict, json_path = json_config_file
            action = ConfigAction(option_strings=None, dest="input")
            namespace: Configuration = arg_parser().parse_args([])
            action(None, namespace, json_path, None)

            # Check that values were loaded correctly
            assert namespace.verbose == json_dict["verbose"]
            assert namespace.seed == json_dict["seed"]
            assert namespace.system_dynamics is not None
            assert isinstance(namespace.X_bounds, RectSet)
            assert isinstance(namespace.X_init, RectSet)
            assert isinstance(namespace.X_unsafe, RectSet)
            assert namespace.gamma == json_dict["gamma"]
            assert namespace.c_coefficient == json_dict["c_coefficient"]
            assert namespace.epsilon == json_dict["epsilon"]
            assert namespace.b_kappa == json_dict["b_kappa"]
            assert namespace.b_norm == json_dict["b_norm"]
            assert namespace.lambda_ == json_dict["lambda"]
            assert namespace.sigma_f == json_dict["sigma_f"]
            assert np.array_equal(namespace.sigma_l, json_dict["sigma_l"])
            assert namespace.feature_map == ConstantTruncatedFourierFeatureMap

            assert namespace.noise_scale == json_dict["noise_scale"]
            assert not namespace.plot
            assert not namespace.verify
