import logging
from inspect import iscoroutinefunction, ismethod
from typing import Any, Callable, Generic, TypeVar
from weakref import ReferenceType, WeakMethod, WeakSet, ref

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


class Signal(Generic[_T]):
    _listeners: WeakSet[WeakMethod | ReferenceType]

    def __init__(self) -> None:
        self._listeners = WeakSet()

    def listen(self, listener: Callable[[_T], Any]) -> None:
        logger.debug("adding listener %s", listener.__qualname__)
        if ismethod(listener):
            self._listeners.add(WeakMethod(listener))
            return
        self._listeners.add(ref(listener))

    async def emit(self, payload: _T) -> None:
        for listener in self._listeners:
            func = listener()
            if func is None:
                continue
            if iscoroutinefunction(func):
                await func(payload)
            else:
                func(payload)
