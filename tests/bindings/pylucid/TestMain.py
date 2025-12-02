from pylucid import __doc__, __version__


class TestMain:
    def test_version(self):
        assert __version__ == "0.0.2"

    def test_docstring(self):
        assert __doc__ == "Lifting-based Uncertain Control Invariant Dynamics"
