"""pylucid bindings for Python."""

from importlib.util import find_spec

from ._constants import GUROBI_BUILD, ALGLIB_BUILD, HIGHS_BUILD, SOPLEX_BUILD, MATPLOTLIB_BUILD, OMP_BUILD, CUDA_BUILD


if GUROBI_BUILD:
    import gurobipy as _gurobipy
from ._pylucid import *
from ._pylucid import __doc__ as __pylucid_doc__
from ._pylucid import __version__ as __pylucid_version__
from .cli import *
from .ext import *
from .parser import *
from .util import assert_or_raise, raise_error

__version__ = __pylucid_version__
__doc__ = __pylucid_doc__

CAPABILITIES = {
    "GUROBI": GUROBI_BUILD,
    "ALGLIB": ALGLIB_BUILD,
    "HIGHS": HIGHS_BUILD,
    "SOPLEX": SOPLEX_BUILD,
    "MATPLOTLIB": MATPLOTLIB_BUILD,
    "PLOT": find_spec("plotly") is not None,
    "VERIFICATION": find_spec("dreal") is not None,
    "GUI": find_spec("flask") is not None,
    "OMP": OMP_BUILD,
}

# Initial verbosity level. Can be changed later.
log.set_verbosity(log.LOG_INFO)
