import threading

stop_event = threading.Event()


def stop_requested() -> bool:
    """True once an external event has asked for a cooperative stop."""
    return stop_event.is_set()


def request_stop() -> None:
    """Ask every hot loop to stop as soon as possible (idempotent)."""
    stop_event.set()
