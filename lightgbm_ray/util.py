from contextlib import closing
import socket
import errno

from lightgbm.basic import _safe_call


class lgbm_network_free:
    """Context to ensure LGBM_NetworkFree() is called
    (makes sure network is cleaned and ports are
    opened even if training fails)."""

    def __init__(self, model, lib) -> None:
        self.model = model
        self.lib = lib
        return

    def __enter__(self) -> None:
        return

    def __exit__(self, type, value, traceback):
        try:
            self.model._Booster.free_network()
        except Exception:
            pass
        _safe_call(self.lib.LGBM_NetworkFree())


def find_free_port() -> int:
    """Find random free port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def is_port_free(port: int) -> bool:
    """Check if port is free"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        try:
            s.bind(("", port))
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                return False
            raise e
    return True
