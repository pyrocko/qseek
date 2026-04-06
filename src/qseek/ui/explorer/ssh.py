from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
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
        self.created = created

        self.updated = asyncio.Event()

        self._tempfolder = tempfile.TemporaryDirectory(prefix="qseek-ssh-")
        logger.debug(
            "Created temporary folder at %s for run %s",
            self._tempfolder.name,
            self.name,
        )
        # self._task = asyncio.create_task(self.watch_for_updates())

    @classmethod
    async def create(cls, connection: asyncssh.SSHClientConnection, remote_path: Path):
        search_json = remote_path / "search.json"
        detections_json = remote_path / "detections.json"
        result = await connection.run(f"sha1sum {search_json}", check=True)
        hash = result.stdout.strip().split()[0]
        result = await connection.run(f"stat -c %Y {detections_json}", check=True)
        created = datetime.fromtimestamp(int(result.stdout.strip()))  # noqa
        result = await connection.run(
            f"wc -l {detections_json}",
            check=True,
        )
        n_events = int(result.stdout.strip().split()[0])
        return cls(connection, remote_path, n_events, hash, created)

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
        while True:
            await asyncio.sleep(poll_interval)
            result = await self.connection.run(
                f"stat -c %Y {self.remote_path / 'detections.json'}",
                check=True,
            )
            modified = datetime.fromtimestamp(int(result.stdout.strip()))  # noqa
            if modified > self.created:
                logger.info("Run %s has been updated, refreshing...", self.name)
                result = await self.connection.run(
                    f"wc -l {self.remote_path / 'detections.json'}",
                    check=True,
                )

                self.n_events = int(result.stdout.strip().split()[0])
                self.created = modified
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
        if self._connection is None:
            self._connection = await asyncssh.connect(
                self.ssh.host,
                port=int(self.ssh.port or 22),
                compression_algs="zlib@openssh.com",
                username=self.ssh.userinfo.split(":")[0]
                if self.ssh.userinfo
                else os.getlogin(),
            )

        path = self.ssh.path
        if path.startswith("/~"):
            path = path.replace("/~", "~/", 1)
        result = await self._connection.run(
            f"""
python3 -c "from pathlib import Path;
directory = Path('{path}').expanduser();
[print(str(p.parent)) for p in directory.glob('*/search.json')];"
""",
            check=True,
        )
        for line in result.stdout.splitlines():
            yield await SshSource.create(self._connection, Path(line))

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
                f"Found run created at {run.created} with {run.n_events} events"
            )

    asyncio.run(main())
