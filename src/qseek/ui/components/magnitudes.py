from qseek.ui.base import Component


class MagnitudeFrequency(Component):
    name = "Magnitude Frequency"
    description = """Frequency of detected events over magnitude bins."""

    async def view(self) -> None: ...


class MagnitudeSemblance(Component):
    name = "Magnitude vs Semblance"
    description = """Magnitude of detected events over their semblance value."""

    async def view(self) -> None: ...
