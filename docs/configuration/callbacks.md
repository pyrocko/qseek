---
icon: lucide/bell
---

[](){ #qseek.plugins.CallbackType }

# Callbacks

Callbacks run your code at defined points of a search, its hooks, e.g. to send an alert for every new detection during real-time monitoring. `callbacks` is a list of callbacks; Qseek calls each of them:

| Hook | When |
| --- | --- |
| `on_start` | the search starts |
| `on_batch_start`, `on_batch_end` | before and after each window of waveforms |
| `on_new_detection` | for every new detection, after its magnitudes and features |
| `on_stop` | the search ends |

## Telegram alerts

Sends detection alerts to a Telegram chat. Set the bot token with the `QSEEK_TELEGRAM_BOT_TOKEN` environment variable: it is not stored in the run's `search.json`. Set the chat with `chat_id` in the configuration or with the `QSEEK_TELEGRAM_CHAT_ID` environment variable. The [real-time monitoring guide](../guides/real-time.md#send-alerts) shows the setup.

```python exec='on'
from qseek.utils import json_example
from qseek.plugins.telegram import TelegramAlert

print(json_example(TelegramAlert.model_construct()))
```

<div class="qs-config" markdown>

::: qseek.plugins.telegram.TelegramAlert
    options:
      heading_level: 3

</div>

## Custom callbacks

Write your own callback in a single Python file and list it in [`callback_scripts`][qseek.search.Search.callback_scripts]. The file subclasses [`Callback`][qseek.plugins.callback.Callback], implements the hooks it needs, and defines a top-level `load()` function that returns an instance.

```python title="my_callback.py"
from qseek.plugins import Callback


class PrintDetections(Callback):
    async def on_new_detection(self, detection):
        print(f"new event at {detection.time}, semblance {detection.semblance:.2f}")


def load() -> Callback:
    return PrintDetections()
```
