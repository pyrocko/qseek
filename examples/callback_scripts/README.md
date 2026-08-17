# Callback Scripts

Single-file plugins that hook into the search lifecycle, without needing to
install a package. Point qseek at one or more `.py` files from the search
config and it will run your code as new batches are processed and new events
are detected.

## Writing a callback script

A callback script is a plain `.py` file that:

1. Defines a subclass of `qseek.plugins.callback.Callback`, overriding
   whichever async hooks it needs.
2. Defines a top-level `load()` function that returns an instance of that
   class.

```python
from qseek.plugins.callback import Callback


class MyCallback(Callback):
    async def on_new_detection(self, detection):
        print(f"new event: {detection.uid} at {detection.time}")


def load() -> Callback:
    return MyCallback()
```

Available hooks (all optional, all async, all no-ops by default):

| Hook | Called |
| --- | --- |
| `on_start(search)` | once, after the search has been prepared |
| `on_batch_start(batch)` | before a waveform batch is processed |
| `on_batch_end(batch)` | after a waveform batch has been processed |
| `on_new_detection(detection)` | for every new event detection |
| `on_stop(search)` | once, after the search has finished |

Notes:

- The file is loaded with its path resolved relative to the current working
  directory qseek is run from.
- A hook that raises is caught and logged; it does not abort the search. A
  broken callback degrades to "no notifications", not "no detections".
- Hooks are `async`. Prefer a library with a native async API (e.g.
  `psycopg.AsyncConnection`, `aiohttp`). If your code only offers a blocking
  call (a synchronous DB driver, `requests`, etc.), run it with
  `await asyncio.to_thread(...)` so it doesn't stall the search loop.

## Enabling a script

Add its path to `callback_scripts` in the search config:

```json
{
  "callback_scripts": ["examples/callback_scripts/postgres_callback.py"]
}
```

Installed callback plugins (shipped with qseek or a separate package) go in
the `callbacks` field instead, configured like any other pluggable component
(`{"callback": "...", ...}`).

## Examples in this folder

- [`postgres_callback.py`](postgres_callback.py) — inserts every new
  detection into a Postgres database via `psycopg`'s async API, reusing one
  connection across calls with an `asyncio.Lock` and a single
  reconnect-and-retry on failure. Connection settings are `pydantic-settings`
  fields on the callback itself (`database`, `producer_user`), read from
  `QSEEKV2_DATABASE` / `QSEEKV2_PRODUCER_USER`. Run it directly
  (`python postgres_callback.py`) to check connectivity and role
  permissions before wiring it into a search. Requires `psycopg` and
  `pydantic-settings`.
