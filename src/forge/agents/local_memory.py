"""Conservative Windows headroom check before loading a local language model."""

import ctypes
import sys

from forge.agents.ollama_client import OllamaError


def windows_memory():
    if sys.platform != "win32":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_commit", ctypes.c_ulonglong),
            ("available_commit", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OllamaError("Windows memory availability could not be checked.")
    return {
        "available_ram_gib": status.available_physical / 1024**3,
        "available_commit_gib": status.available_commit / 1024**3,
    }


def require_memory_headroom():
    memory = windows_memory()
    if memory is not None and (
        memory["available_ram_gib"] < 6 or memory["available_commit_gib"] < 8
    ):
        raise OllamaError(
            "Local model not started: this demo requires at least 6 GiB of free RAM "
            "and 8 GiB of Windows commit headroom. Close unused applications first. "
            f"Available: {memory['available_ram_gib']:.1f} GiB RAM, "
            f"{memory['available_commit_gib']:.1f} GiB commit headroom."
        )
    return memory
