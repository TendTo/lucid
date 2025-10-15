import sys

import pytest

if __name__ == "__main__":
    sys.path.insert(0, "bindings")

    import pylucid

    pylucid.log.set_verbosity(0)  # disable logging for tests

    sys.exit(pytest.main(sys.argv[1:]))
