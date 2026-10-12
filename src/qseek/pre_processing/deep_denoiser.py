from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from obspy import Stream
from pydantic import Field, PrivateAttr
from pyrocko import obspy_compat

from qseek.pre_processing.base import BatchPreProcessing

if TYPE_CHECKING:
    from seisbench.models import DeepDenoiser as SeisBenchDeepDenoiser

    from qseek.waveforms.base import WaveformBatch


logger = logging.getLogger(__name__)
DenoiserModels = Literal["original", "instance"]
obspy_compat.plant()


class DeepDenoiser(BatchPreProcessing):
    """De-noise the traces using the DeepDenoiser neural network (slow)."""

    process: Literal["deep-denoiser"] = "deep-denoiser"

    model: DenoiserModels = Field(
        "original",
        description="The model to use for denoising.",
    )
    torch_use_cuda: bool | str = Field(
        False,
        description="Use CUDA for the PyTorch model. A string selects the device.",
    )

    _denoiser: SeisBenchDeepDenoiser = PrivateAttr()

    async def prepare(self) -> None:
        import torch
        from seisbench.models import DeepDenoiser as SeisBenchDeepDenoiser

        self._denoiser = SeisBenchDeepDenoiser.from_pretrained(self.model)
        if self.torch_use_cuda:
            if isinstance(self.torch_use_cuda, bool):
                self._denoiser.cuda()
            else:
                self._denoiser.cuda(self.torch_use_cuda)

        self._denoiser.eval()
        try:
            logger.info("compiling DeepDenoiser model...")
            self._denoiser = torch.compile(self._denoiser, mode="max-autotune")
        except RuntimeError as exc:
            logger.warning(
                "failed to compile PhaseNet model, using uncompiled model.",
                exc_info=exc,
            )

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        if self._denoiser is None:
            raise RuntimeError("DeepDenoiser model not initialized.")

        traces = self.filter_traces(batch)
        if not traces:
            return batch
        stream = Stream(tr.to_obspy_trace() for tr in traces)
        stream = await self._denoiser.annotate_async(stream)

        denoised_traces = []
        for tr in stream:
            trace = tr.to_pyrocko_trace()
            trace.set_codes(channel=trace.channel.removeprefix("DeepDenoiser_"))
            denoised_traces.append(trace)

        # The denoised traces replace the traces they were computed from
        replaced = {id(tr) for tr in traces}
        batch.traces = [
            tr for tr in batch.traces if id(tr) not in replaced
        ] + denoised_traces
        return batch
