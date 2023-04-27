from typing import Literal

from lassie.tracers.base import RayTracer


class CakeTracer(RayTracer):
    tracer: Literal["CakeTracer"] = "CakeTracer"
