from contextlib import closing
import socket
import errno
import traceback as tb

from lightgbm.basic import _safe_call


class lgbm_network_free:
    def __init__(self, model, init_model, lib) -> None:
        """Context to ensure free_network() is called"""
        self.model = model
        self.init_model = init_model
        self.lib = lib
        return

    def __enter__(self) -> None:
        _safe_call(self.lib.LGBM_NetworkFree())

    def __exit__(self, type, value, traceback):
        try:
            self.init_model.free_network()
        except Exception as e:
            pass
        try:
            self.model._Booster.free_network()
        except Exception as e:
            pass
        _safe_call(self.lib.LGBM_NetworkFree())


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def is_port_free(port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        try:
            s.bind(("", port))
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                return False
            raise e
    return True
