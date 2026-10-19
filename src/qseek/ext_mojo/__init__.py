"""Mojo extension modules, compiled on demand by the `mojo.importer` hook.

Importing a name from this package (e.g. `from qseek.ext_mojo import
delay_sum`) compiles the matching `.mojo` source into `__mojocache__/` on
first use and reuses it until the source changes.
"""

from __future__ import annotations
