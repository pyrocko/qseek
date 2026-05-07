# Extras

## systemd Service (`qseek.service`)

Runs qseek as a managed system service with watchdog monitoring. The service uses `Type=notify`, so systemd tracks readiness and health directly from the process. If qseek stops petting the watchdog for more than 30 seconds (e.g. due to a data stall), systemd kills and restarts it automatically.

### Configuration

Before installing, edit `qseek.service` and set:

- **`WorkingDirectory`** — the directory containing your `project.json`
- **`ExecStart`** — the path to the qseek binary in your environment (see comments in the file for uv and conda variants)

### Installation

Copy the unit file to the systemd user or system directory:

```bash
# System-wide (requires root)
sudo cp qseek.service /etc/systemd/system/

# Or per-user (no root required)
cp qseek.service ~/.config/systemd/user/
```

Reload systemd and enable the service:

```bash
# System-wide
sudo systemctl daemon-reload
sudo systemctl enable --now qseek.service

# Per-user
systemctl --user daemon-reload
systemctl --user enable --now qseek.service
```

### Common commands

```bash
systemctl status qseek        # show current status and recent log lines
journalctl -u qseek -f        # follow live logs
systemctl stop qseek          # graceful stop
systemctl restart qseek       # restart immediately
```

For per-user installs, prepend `--user` to each command.
