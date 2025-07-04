"""pylucid bindings for Python."""

import os

if os.name == "nt" and os.environ.get("GUROBI_HOME", "") != "":
    # Windows
    os.add_dll_directory(os.path.join(os.environ.get("GUROBI_HOME", ""), "bin"))

from ._pylucid import *
from ._pylucid import __doc__ as __pylucid_doc__
from ._pylucid import __version__ as __pylucid_version__
from .cli import *
from .ext import *
from .parser import *
from .util import raise_error, assert_or_raise

__version__ = __pylucid_version__
__doc__ = __pylucid_doc__


# Initial verbosity level. Can be changed later.
log.set_verbosity(log.LOG_INFO)
