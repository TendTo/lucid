"""pylucid bindings for Python."""

import os

if os.name == "nt":
    # Windows
    os.add_dll_directory(f'{os.environ.get("GUROBI_HOME", "")}/bin')

from ._pylucid import *
from ._pylucid import __version__ as __pylucid_version__, __doc__ as __pylucid_doc__

__version__ = __pylucid_version__
__doc__ = __pylucid_doc__

# Initial verbosity level. Can be changed later.
set_verbosity(LOG_INFO)
