"""pylucid bindings for Python."""

from importlib.util import find_spec

from ._constants import (
    ALGLIB_BUILD,
    CUDA_BUILD,
    GUROBI_BUILD,
    HIGHS_BUILD,
    MATPLOTLIB_BUILD,
    OMP_BUILD,
    PSOCPP_BUILD,
    SOPLEX_BUILD,
)

if GUROBI_BUILD:
    import sys as _sys

    if _sys.version_info >= (3, 9):  # Python 3.9, use gurobipy wheel with embedded shared libraries
        try:
            import gurobipy as _gurobipy
        except ImportError as e:
            raise ImportError("Could not import gurobipy. Make sure it is installed with 'pip install gurobipy'") from e
    else:  # Older Python versions, use system-installed Gurobi, if available
        import os as _os

        if _os.name == "nt" and _os.environ.get("GUROBI_HOME", "") != "":
            # Windows
            _os.add_dll_directory(_os.path.join(_os.environ.get("GUROBI_HOME", ""), "bin"))

if CUDA_BUILD:
    import os as _os

    try:
        from cuda.pathfinder import load_nvidia_dynamic_lib as _load_nvidia_dynamic_lib
    except ImportError as e:
        raise ImportError(
            "Could not import cuda-pathfinder. Make sure it is installed with 'pip install cuda-pathfinder'"
        ) from e

    _os.add_dll_directory(_os.path.dirname(_load_nvidia_dynamic_lib("cublas").abs_path))

from ._pylucid import *
from ._pylucid import __doc__ as __pylucid_doc__
from ._pylucid import __version__ as __pylucid_version__
from .cli import (
    ConfigAction,
    Configuration,
    EstimatorAction,
    FeatureMapAction,
    FloatOrNVectorAction,
    KernelAction,
    MultiNMatrixAction,
    NMatrixAction,
    OptimiserAction,
    SystemDynamicsAction,
    arg_parser,
)
from .parser import DrealParser, SetParser, SymbolicParser, SympyParser, Z3Parser
from .util import assert_or_raise, raise_error

__version__ = __pylucid_version__
__doc__ = __pylucid_doc__

CAPABILITIES = {
    "GUROBI": GUROBI_BUILD,
    "ALGLIB": ALGLIB_BUILD,
    "HIGHS": HIGHS_BUILD,
    "SOPLEX": SOPLEX_BUILD,
    "PSOCPP": PSOCPP_BUILD,
    "MATPLOTLIB": MATPLOTLIB_BUILD,
    "PLOT": find_spec("plotly") is not None,
    "VERIFICATION": find_spec("dreal") is not None,
    "GUI": find_spec("flask") is not None,
    "OMP": OMP_BUILD,
    "CUDA": CUDA_BUILD,
}

# Initial verbosity level. Can be changed later.
log.set_verbosity(log.LOG_INFO)
