"""Tests for the provenance path helper.

Both sides must be resolved: inside a Nextflow task ``/results`` and
``/data`` are symlinks into the work directory, and comparing a resolved
root against an unresolved path leaks an absolute container path into the
emitted metadata.
"""

import os
import tempfile
import unittest
from pathlib import Path

from aind_ophys_nwb_library.utils.paths import (
    relative_to_root,
)


class TestRelativeToRoot(unittest.TestCase):
    """Relative provenance paths, including the symlinked-root case."""

    def test_plain_relative_path(self):
        """A path under the root is returned relative to it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "nwb"
            target.mkdir(parents=True)
            self.assertEqual(relative_to_root(root, target), "nwb")

    def test_symlinked_root_still_resolves(self):
        """A symlinked root does not leak an absolute path."""
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "work"
            (real / "nwb").mkdir(parents=True)
            link = Path(tmp) / "results"
            os.symlink(real, link)
            self.assertEqual(
                relative_to_root(link, link / "nwb"),
                "nwb",
            )

    def test_path_outside_root_falls_back_to_absolute(self):
        """A path genuinely outside the root keeps full provenance."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            root.mkdir()
            outside = Path(tmp) / "elsewhere"
            outside.mkdir()
            self.assertEqual(relative_to_root(root, outside), str(outside))


if __name__ == "__main__":
    unittest.main()
