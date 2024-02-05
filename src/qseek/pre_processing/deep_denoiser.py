from __future__ import annotations

import asyncio
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
    """De-noise the traces using the DeepDenoiser neural network (Zhu et al., 2019)."""

    process: Literal["deep-denoiser"] = "deep-denoiser"

    model: DenoiserModels = Field(
        "original",
        description="The model to use for denoising.",
    )
    torch_use_cuda: bool | str = Field(
        False,
        description="Whether to use CUDA for the PyTorch model."
        "A string can be used to specify the device.",
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

        stream = Stream(tr.to_obspy_trace() for tr in self.select_traces(batch))
        stream = await asyncio.to_thread(self._denoiser.annotate, stream)

        denoised_traces = [tr.to_pyrocko_trace() for tr in stream]
        for tr in denoised_traces:
            tr.channel = tr.channel.replace("DeepDenoiser_", "")

        return batch
