---
icon: lucide/radar
---

# Real-time monitoring

Qseek detects and locates earthquakes in real time from SeedLink streams, and sends alerts for new detections. The setup is the same as for an archive search; only the data provider and the callbacks change.

## Stream the waveforms

Replace the `data_provider` with the [SeedLink provider](../configuration/waveforms.md#seedlink). Every client connects to one SeedLink server and requests the listed stations:

```json title="SeedLink provider"
"data_provider": {
  "provider": "SeedLink",
  "clients": [
    {
      "host": "geofon.gfz.de",
      "port": 18000,
      "station_selection": [
        {"nsl": "GE.RUE.", "channel": "HH?"}
      ]
    }
  ],
  "sds_archive": "data/sds"
}
```

With [`sds_archive`][qseek.waveforms.seedlink.seedlink.SeedLink.sds_archive], Qseek also archives the received waveforms, so you can search them again later, e.g. with station corrections. The SeedLink provider needs `slinktool`, see [SeedLink](../configuration/waveforms.md#seedlink).

## Send alerts

Add a [Telegram alert](../configuration/callbacks.md#telegram-alerts) to `callbacks`. It notifies a chat about detections above [`magnitude_alert`][qseek.plugins.telegram.TelegramAlert.magnitude_alert], and about swarms: at least [`rate_alert_count`][qseek.plugins.telegram.TelegramAlert.rate_alert_count] events at or above [`rate_alert_magnitude`][qseek.plugins.telegram.TelegramAlert.rate_alert_magnitude] within [`rate_alert_window`][qseek.plugins.telegram.TelegramAlert.rate_alert_window].

```json title="Telegram alerts"
"callbacks": [
  {
    "callback": "TelegramAlert",
    "magnitude_alert": 2.0
  }
]
```

Set the bot token in the environment, not in the configuration, so it is not written into the run directory. Set the chat ID there too, or as `chat_id` in the configuration:

```sh
export QSEEK_TELEGRAM_BOT_TOKEN="<token from BotFather>"
export QSEEK_TELEGRAM_CHAT_ID="<chat id>"
qseek search monitoring.json
```

For other alerts, write a [custom callback](../configuration/callbacks.md#custom-callbacks).

## Watch the search

- **Web server:** the [`webserver`][qseek.search.Search.webserver] of the search serves the detections and the state of the running search over HTTP while it runs.
- **Web UI:** explore the detections as they come in with [`qseek explore`](../results/explore.md#web-ui), also on a remote machine through SSH.

## Run as a service

Qseek reports its state to systemd through `sd_notify`, so you can run the monitoring as a systemd service of `Type=notify` that systemd restarts when it stops. The [service file template](https://github.com/pyrocko/qseek/blob/main/extras/qseek.service) can be found in the GitHub repository.
