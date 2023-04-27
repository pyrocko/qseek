from typing import Literal

from lassie.tracers.base import Tracer


class CakeTracer(Tracer):
    tracer: Literal["CakeTracer"] = "CakeTracer"
