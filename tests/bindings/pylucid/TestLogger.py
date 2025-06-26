from pylucid import log


class TestLogger:
    def test_log_trace(self):
        assert log.trace("This is a trace message") is None

    def test_log_debug(self):
        assert log.debug("This is a debug message") is None

    def test_log_info(self):
        assert log.info("This is an info message") is None

    def test_log_warn(self):
        assert log.warn("This is a warning message") is None

    def test_log_error(self):
        assert log.error("This is an error message") is None

    def test_log_critical(self):
        assert log.critical("This is a critical message") is None
