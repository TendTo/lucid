from ._pylucid import log_error


def raise_error(message: str, error_type: type = ValueError):
    """Raise an error with the given message."""
    log_error(message)
    return error_type(message)
