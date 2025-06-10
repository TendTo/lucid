from pylucid import log_trace, log_debug, log_info, log_warn, log_error, log_critical


class TestLogger:
    def test_log_trace(self):
        assert log_trace("This is a trace message") is None

    def test_log_debug(self):
        assert log_debug("This is a debug message") is None

    def test_log_info(self):
        assert log_info("This is an info message") is None

    def test_log_warn(self):
        assert log_warn("This is a warning message") is None

    def test_log_error(self):
        assert log_error("This is an error message") is None

    def test_log_critical(self):
        assert log_critical("This is a critical message") is None
