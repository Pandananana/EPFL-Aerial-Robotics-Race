"""Cross-thread communication primitives.

Two patterns are used in this codebase:

1. Qt signals (e.g. `frame_ready = QtCore.pyqtSignal(object)`) — for events
   where every consumer should react to every message. Qt's auto-queued
   connection across threads is built in; no extra locking needed.

2. `Latest[T]` (this module) — for "what is the most recent value of X?"
   Producers call `latch.set(value)`; consumers call `latch.get()`. Old
   values are dropped, which is the right default for real-time control
   loops that poll state at their own rate.

Rule of thumb:
  - "I need every message"            -> Qt signal
  - "I need the most recent value"    -> Latest[T]
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class Latest(Generic[T]):
    """Thread-safe single-slot latch. Reads return the most recently set value,
    or None if nothing has been set yet."""

    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: T | None = None

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value

    def get(self) -> T | None:
        with self._lock:
            return self._value
