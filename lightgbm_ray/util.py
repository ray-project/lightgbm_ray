from typing import Dict, Optional, List

from contextlib import closing
import socket

from lightgbm.basic import _safe_call

class lgbm_network_free:
    def __init__(self, lib) -> None:
        """Context to ensure LGBM_NetworkFree() is called"""
        self.lib = lib
        return

    def __enter__(self) -> None:
        return

    def __exit__(self, type, value, traceback):
        _safe_call(self.lib.LGBM_NetworkFree())

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]