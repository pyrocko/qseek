from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import tempfile
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

import asyncssh
import rfc3986

from qseek.ui.explorer.base import RunExplorer, RunSource

logger = logging.getLogger(__name__)


class SshSource(RunSource):
    source = "ssh"

    def __init__(
        self,
        connection: asyncssh.SSHClientConnection,
        remote_path: Path,
        n_events: int,
        hash: str,
        created: datetime,
    ):
        self.name = remote_path.name
        self.remote_path = remote_path
        self.connection = connection
        self.hash = hash

        self.n_events = n_events
        self.last_update = created

        self.updated = asyncio.Event()

        self._tempfolder = tempfile.TemporaryDirectory(prefix="qseek-ssh-")
        logger.debug(
            "Created temporary folder at %s for run %s",
            self._tempfolder.name,
            self.name,
        )
        # self._task = asyncio.create_task(self.watch_for_updates())

    @classmethod
    def from_metadata(
        cls,
        connection: asyncssh.SSHClientConnection,
        remote_path: Path,
        hash: str,
        mtime: int,
        n_events: int,
    ) -> SshSource:
        """Construct from pre-fetched metadata (avoids extra SSH channels)."""
        created = datetime.fromtimestamp(mtime)  # noqa: DTZ006
        return cls(connection, remote_path, n_events, hash, created)

    @classmethod
    async def create(cls, connection: asyncssh.SSHClientConnection, remote_path: Path):
        """Create by fetching metadata in a single SSH channel."""
        search_json = remote_path / "search.json"
        detections_json = remote_path / "detections.json"

        result = await connection.run(
            f'python3 -c "'
            f"import hashlib, os; "
            f"h = hashlib.sha1(open('{search_json}', 'rb').read()).hexdigest(); "
            f"s = os.stat('{detections_json}'); "
            f"n = sum(1 for _ in open('{detections_json}')); "
            f"print(h, int(s.st_mtime), n)"
            f'"',
            check=True,
        )
        parts = result.stdout.strip().split()
        hash, mtime, n_events = parts[0], int(parts[1]), int(parts[2])
        return cls.from_metadata(connection, remote_path, hash, mtime, n_events)

    async def get_search_json(self) -> Path:
        await asyncssh.scp(
            (self.connection, str(self.remote_path / "search.json")),
            self._tempfolder.name,
        )
        return Path(self._tempfolder.name) / "search.json"

    async def _copy_catalog_files(self, force: bool = False) -> None:
        detections_json = Path(self._tempfolder.name) / "detections.json"
        receivers_json = Path(self._tempfolder.name) / "detections_receivers.json"

        if not force and detections_json.exists() and receivers_json.exists():
            logger.debug(
                "Catalog files already exist in %s, skipping copy",
                self._tempfolder.name,
            )
            return
        await asyncssh.scp(
            (self.connection, str(self.remote_path / "detections.json")),
            self._tempfolder.name,
        )
        await asyncssh.scp(
            (self.connection, str(self.remote_path / "detections_receivers.json")),
            self._tempfolder.name,
        )
        logger.info("Catalog files copied to %s", self._tempfolder.name)

    async def get_catalog_path(self) -> Path:
        await self._copy_catalog_files(force=False)
        return Path(self._tempfolder.name)

    async def watch_for_updates(self, poll_interval: float = 60.0):
        detections_json = self.remote_path / "detections.json"
        while True:
            await asyncio.sleep(poll_interval)
            # Fetch mtime and line count in a single SSH channel
            result = await self.connection.run(
                f'python3 -c "'
                f"import os; "
                f"s = os.stat('{detections_json}'); "
                f"n = sum(1 for _ in open('{detections_json}')); "
                f"print(int(s.st_mtime), n)"
                f'"',
                check=True,
            )
            parts = result.stdout.strip().split()
            modified = datetime.fromtimestamp(int(parts[0]))  # noqa: DTZ006
            if modified > self.last_update:
                logger.info("Run %s has been updated, refreshing...", self.name)
                self.n_events = int(parts[1])
                self.last_update = modified
                await self._copy_catalog_files(force=True)

                self.updated.set()
                self.updated.clear()

    def __del__(self):
        self._tempfolder.cleanup()


class SshExplorer(RunExplorer):
    _connection: asyncssh.SSHClientConnection | None = None

    def __init__(self, ssh_uri: str):
        """Initialize the SshExplorer with the given SSH URI.

        Args:
            ssh_uri (str): The SSH URI to connect to.
            E.g. "ssh://user@host:port/path/to/runs".

        Raises:
            ValueError: If the SSH URI is invalid.
        """
        ssh = rfc3986.urlparse(ssh_uri)
        if ssh.scheme != "ssh":
            raise ValueError("Invalid SSH URI: scheme must be 'ssh'")
        if not ssh.path:
            raise ValueError("Invalid SSH URI: path is required")

        self.ssh = ssh

    async def discover(self) -> AsyncIterator[RunSource]:
        options = {}
        if self.ssh.userinfo:
            parts = self.ssh.userinfo.split(":")
            options["username"] = parts[0]
            if len(parts) == 2:
                options["password"] = parts[1]

        if self._connection is None:
            self._connection = await asyncssh.connect(
                self.ssh.host,
                port=int(self.ssh.port or 22),
                compression_algs="zlib@openssh.com",
                **options,
            )

        path = self.ssh.path
        if path.startswith("/~"):
            path = path.replace("/~", "~/", 1)
        # One SSH channel to get all run metadata at once
        result = await self._connection.run(
            f'python3 -c "'
            f"import hashlib, os; "
            f"from pathlib import Path; "
            f"directory = Path('{path}').expanduser(); "
            f"runs = [p.parent for p in directory.glob('*/search.json')]; "
            f"[print(str(r), "
            f"hashlib.sha1(open(r/'search.json','rb').read()).hexdigest(), "
            f"int(os.stat(r/'detections.json').st_mtime), "
            f"sum(1 for _ in open(r/'detections.json'))) "
            f"for r in runs if (r/'detections.json').exists()]"
            f'"',
            check=True,
        )
        for line in result.stdout.splitlines():
            try:
                rundir, hash, mtime, n_events = line.split()
                yield SshSource.from_metadata(
                    self._connection,
                    Path(rundir),
                    hash,
                    int(mtime),
                    int(n_events),
                )
            except Exception as e:
                logger.warning("Failed to create SshSource for %s: %s", line, e)

    def __del__(self):
        if self._connection:
            with contextlib.suppress(RuntimeError):
                self._connection.close()
            self._connection = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover Qseek runs over SSH")
    parser.add_argument("ssh_uri", type=str, help="SSH URI to connect to")
    args = parser.parse_args()

    discoverer = SshExplorer(args.ssh_uri)

    async def main():
        logging.basicConfig(level=logging.DEBUG)
        async for run in discoverer.discover():
            print(  # noqa
                f"Found run created at {run.last_update} with {run.n_events} events"
            )

    asyncio.run(main())
