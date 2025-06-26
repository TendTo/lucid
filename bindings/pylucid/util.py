from ._pylucid import log


def raise_error(message: str, error_type: type = ValueError):
    """Raise an error with the given message."""
    log.error(message)
    return error_type(message)
