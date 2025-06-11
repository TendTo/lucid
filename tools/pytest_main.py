import sys

import pytest

if __name__ == "__main__":
    sys.path.insert(0, "bindings")
    sys.exit(pytest.main(sys.argv[1:]))
