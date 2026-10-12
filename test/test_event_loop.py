from __future__ import annotations

import subprocess
import sys
import textwrap

# Runs the code path of the CLI in a fresh interpreter: `asyncio.run()` without
# nest_asyncio, a task group and timeouts in the main task, and SeisBench, whose
# synchronous `annotate()` calls `asyncio.run()` itself.
CLI_EVENT_LOOP = textwrap.dedent(
    """
    import asyncio

    import numpy as np
    import seisbench.models as sbm
    from pyrocko.trace import Trace

    import qseek.apps.qseek  # noqa: F401
    from qseek.images.seisbench import SeisBench
    from qseek.utils import BackgroundTasks


    async def main() -> None:
        assert asyncio.current_task() is not None

        function = SeisBench(model="PhaseNet")
        function._seisbench_model = sbm.PhaseNet()  # random weights, no download
        rng = np.random.default_rng(0)
        traces = [
            Trace(
                network="XX",
                station="STA",
                channel=f"HH{component}",
                tmin=1700000000.0,
                deltat=0.01,
                ydata=rng.normal(size=6000),
            )
            for component in "ZNE"
        ]

        async with asyncio.timeout(60.0), BackgroundTasks() as tasks:
            task = tasks.create_task(function.process_traces(traces))
        images = task.result()
        assert images is not None
        assert all(image.traces for image in images)

        # Nested event loops fail loudly instead of being patched
        coro = asyncio.sleep(0)
        try:
            asyncio.run(coro)
        except RuntimeError:
            coro.close()
        else:
            raise AssertionError("asyncio.run() nested in a running event loop")


    asyncio.run(main())
    assert not hasattr(asyncio, "_nest_patched"), "nest_asyncio is applied"
    print("ok")
    """
)


def test_cli_event_loop():
    """The search path runs in `asyncio.run()` without nest_asyncio."""
    result = subprocess.run(
        [sys.executable, "-c", CLI_EVENT_LOOP],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
