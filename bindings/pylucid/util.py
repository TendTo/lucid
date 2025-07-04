from ._pylucid import log


def raise_error(message: str, error_type: type = ValueError):
    """Raise an error with the given message."""
    log.error(message)
    return error_type(message)


def assert_or_raise(condition: bool, message: str, error_type: type = ValueError):
    """Assert a condition or raise an error with the given message."""
    if not condition:
        raise raise_error(message, error_type)
