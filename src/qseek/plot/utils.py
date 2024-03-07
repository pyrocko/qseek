from __future__ import annotations

import inspect
from functools import wraps
from typing import Callable, ParamSpec

from matplotlib import pyplot as plt

P = ParamSpec("P")


def with_default_axes(func: Callable[P, None]) -> Callable[P, None]:
    signature = inspect.signature(func)
    if "axes" not in signature.parameters:
        raise AttributeError("Function must have an 'axes' parameter")
    if "filename" not in signature.parameters:
        raise AttributeError("Function must have an 'filename' parameter")

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        axes_provided = kwargs.get("axes") is not None
        if kwargs.get("axes") is None:
            fig = plt.figure()
            ax = fig.gca()
            kwargs["axes"] = ax
        else:
            ax: plt.Axes = kwargs["axes"]
            fig = ax.figure

        ret = func(*args, **kwargs)

        if kwargs.get("filename") is not None:
            fig.savefig(str(kwargs.get("filename")), bbox_inches="tight", dpi=300)
            plt.close()

        if not axes_provided:
            plt.show()
        return ret

    return wrapper
