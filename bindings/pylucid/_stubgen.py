import argparse
import os
import shutil
import sys

from pybind11_stubgen import main as stubgen_main


def generate_stub_files(out_dir: str, _pylucid_pyi: str):
    # We must also add the out_dir, which contains the source files, to the path,
    # so that the pylucid module can be imported
    sys.path = [os.path.abspath(out_dir)] + sys.path
    stubgen_main(("-o", out_dir, "--numpy-array-remove-parameters", "pylucid"))

    # Print all content of the out_dir for debugging purposes
    print(f"Generated stub files in: {out_dir}")
    print("Files in the output directory:")
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            if file.endswith(".pyi"):
                # Only print the generated .pyi files
                print(f"- {os.path.join(root, file)}")

    with open(_pylucid_pyi, "rb") as f:
        content = f.read()
        content = (
            # Imports
            content.replace(
                b"import typing",
                b"import typing\n"
                b"import numpy as np\n"
                b"import numpy.typing as npt\n"
                b"K = typing.TypeVar('K', bound=Kernel)\n"
                b"T = typing.TypeVar('T', bound=Tuner)\n"
                b"NVector = np.ndarray[tuple[int], np.float64]\n"
                b"NMatrix = np.ndarray[tuple[int, int], np.float64]\n"
                b"NArray = npt.NDArray[np.float64]\n"
                b"ParameterValueType = int | float | NVector\n"
                b"ParameterValuesType = tuple[int, ...] | tuple[float, ...] | tuple[NVector, ...]\n",
                1,
            )
            # Generics
            .replace(
                b"class KernelRidgeRegressor(Estimator):",
                b"class KernelRidgeRegressor(Estimator[T], typing.Generic[K, T]):",
                1,
            )
            .replace(
                b"class Estimator(Parametrizable):",
                b"class Estimator(Parametrizable, typing.Generic[T]):",
                1,
            )
            .replace(b"def kernel(self) -> Kernel:", b"def kernel(self) -> K:")
            .replace(
                b"def __init__(self, kernel: Kernel, regularization_constant: float = 1.0, tuner: Tuner = None) -> None:",
                b"def __init__(self, kernel: K, regularization_constant: float = 1.0, tuner: T | None = None) -> None:",
            )
            .replace(b"tuner: Tuner\n", b"tuner: T | None\n")
            # Numpy types
            .replace(b"numpy.ndarray", b"NArray")
            # Clone types
            .replace(b"def clone(self) -> Estimator:", b"def clone(self) -> typing.Self:")
            .replace(b"def clone(self) -> Kernel:", b"def clone(self) -> typing.Self:")
            .replace(b"def clone(self) -> FeatureMap:", b"def clone(self) -> typing.Self:")
            # Parametrizable get method types
            .replace(
                b"def get(self, parameter: Parameter) -> typing.Any:",
                b"def get(self, parameter: Parameter) -> ParameterValueType:",
            )
            .replace(
                b"def value(self) -> typing.Any:",
                b"def value(self) -> ParameterValueType:",
            )
            .replace(
                b"def values(self) -> typing.Any:",
                b"def values(self) -> ParameterValuesType:",
            )
            .replace(
                b"def __init__(self, parameters: dict, n_jobs: int = 0) -> None:",
                b"def __init__(self, parameters: dict[Parameter, ParameterValuesType], n_jobs: int = 0) -> None:",
            )
            .replace(
                b"feature_map_type: type",
                b"feature_map_type: type[TruncatedFourierFeatureMap]",
            )
        )

    with open(_pylucid_pyi, "wb") as f:
        f.write(content)


def generate_pytyped_file(out_file: str):
    with open(out_file, "wb") as f:
        f.write(b"")


def copy_lib(out_dir: str, lib: "list[str]"):
    """Copy the library files from the original position to the execution sandbox

    Args:
        out_dir: output folder. Something like 'bazel-out/k8-fastbuild/bin/bindings'
        lib: list of library files to copy
    """
    sandbox_dir = os.path.join(out_dir, "pylucid")
    for f in lib:
        shutil.copy(f, sandbox_dir)


def main():
    argparser = argparse.ArgumentParser(description="Generate stub files for pylucid")
    argparser.add_argument(
        "-d",
        "--pyd",
        type=str,
        help="Path to the shared object file (e.g., _pylucid.pyd)",
        required=True,
    )
    argparser.add_argument(
        "-s",
        "--so",
        type=str,
        help="Path to the shared object file (e.g., _pylucid.so)",
        required=True,
    )
    argparser.add_argument(
        "-l",
        "--lib",
        type=str,
        nargs="+",
        help="Library files",
        required=True,
    )
    argparser.add_argument(
        "-o",
        "--outs",
        type=str,
        nargs="+",
        help="Output files to generate. Allows multiple files, e.g., _pylucid.pyi, __init__.pyi and py.typed",
        required=True,
    )
    args = argparser.parse_args()
    _pylucid_pyi, _random_pyi, _exception_pyi, _log_pyi, _gurobi_pyi, _init_pyi, py_typed = args.outs
    # The first two are the .pyi files generated by pybind11_stubgen
    # They have a structure like
    # - 'bazel-out/k8-fastbuild/bin/bindings/pylucid/__init__.pyi'
    # - 'bazel-out/k8-fastbuild/bin/bindings/pylucid/_pylucid/__init__.pyi'
    # - 'bazel-out/k8-fastbuild/bin/bindings/pylucid/_pylucid/random.pyi'
    # - 'bazel-out/k8-fastbuild/bin/bindings/pylucid/_pylucid/log.pyi'
    # - 'bazel-out/k8-fastbuild/bin/bindings/pylucid/_pylucid/exception.pyi'
    # We need the path up to 'bin/bindings' to generate the stubs, since that is our output directory
    out_dir = os.path.dirname(os.path.dirname(_init_pyi))

    copy_lib(out_dir, args.lib)
    generate_stub_files(out_dir, _pylucid_pyi)
    generate_pytyped_file(py_typed)

    assert os.path.exists(_pylucid_pyi), f"Output file {_pylucid_pyi} not found"
    assert os.path.exists(_pylucid_pyi), f"Output file {_random_pyi} not found"
    assert os.path.exists(_exception_pyi), f"Output file {_exception_pyi} not found"
    assert os.path.exists(_log_pyi), f"Output file {_log_pyi} not found"
    assert os.path.exists(_gurobi_pyi), f"Output file {_gurobi_pyi} not found"
    assert os.path.exists(_init_pyi), f"Output file {_init_pyi} not found"
    assert os.path.exists(py_typed), f"Output file {py_typed} not found"


if __name__ == "__main__":
    main()
