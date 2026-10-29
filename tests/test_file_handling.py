"""Tests for upstream-artifact discovery over the v1 and v2 layouts."""

import logging
import tempfile
import unittest
from pathlib import Path

from aind_ophys_nwb_library import file_handling

PLANES = ("VISp_0", "VISl_1")

V2_LAYOUT = {
    "extraction": "{plane}_extraction.h5",
    "dff": "{plane}_dff.h5",
    "events": "{plane}_events_oasis.h5",
    "classification": "{plane}_classification.h5",
    "motion_correction": "{plane}_average_projection.png",
}


def _touch(path: Path) -> Path:
    """Create an empty file and every parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _write_v2_plane(root: Path, plane: str, *, complete: bool = True) -> None:
    """Populate one plane's v2 output tree under ``root``."""
    for step, template in V2_LAYOUT.items():
        if not complete and step in ("dff", "events", "classification"):
            continue
        _touch(root / plane / step / template.format(plane=plane))
    _touch(root / plane / "motion_correction" / f"{plane}_maximum_projection.png")


class TestFindArtifact(unittest.TestCase):
    """The single-artifact recursive glob."""

    def test_finds_nested_artifact(self) -> None:
        """A v2 nested path is found by the filename suffix alone."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = _touch(root / "VISp_0" / "dff" / "VISp_0_dff.h5")
            self.assertEqual(
                file_handling.find_artifact(root, "dff.h5", "VISp_0"),
                expected,
            )

    def test_plane_prefix_excludes_other_planes(self) -> None:
        """A plane's glob never resolves to a different plane's file."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "VISl_1" / "dff" / "VISl_1_dff.h5")
            self.assertIsNone(file_handling.find_artifact(root, "dff.h5", "VISp_0"))

    def test_without_plane_prefix_matches_any(self) -> None:
        """Omitting the plane id relaxes the prefix requirement."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = _touch(root / "motion_correction" / "abc_dff.h5")
            self.assertEqual(file_handling.find_artifact(root, "dff.h5"), expected)

    def test_missing_returns_none(self) -> None:
        """No match yields None rather than raising StopIteration."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(file_handling.find_artifact(Path(tmp), "dff.h5", "VISp_0"))

    def test_multiple_matches_warn_and_take_the_first(self) -> None:
        """An ambiguous glob is logged, not silently resolved."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "a" / "VISp_0_dff.h5")
            _touch(root / "b" / "VISp_0_dff.h5")
            with self.assertLogs(file_handling.logger, level=logging.WARNING) as logs:
                found = file_handling.find_artifact(root, "dff.h5", "VISp_0")
            self.assertIn("2 matches", logs.output[0])
            self.assertEqual(found.name, "VISp_0_dff.h5")


class TestCollectPlaneArtifacts(unittest.TestCase):
    """Per-plane artifact resolution against the real v2 layout."""

    def test_complete_v2_plane(self) -> None:
        """Every artifact resolves from the nested v2 layout."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_v2_plane(root, "VISp_0")
            artifacts = file_handling.collect_plane_artifacts(root, "VISp_0")
            self.assertEqual(artifacts.plane_id, "VISp_0")
            self.assertTrue(artifacts.extraction_h5.name.endswith("extraction.h5"))
            self.assertIsNotNone(artifacts.dff_h5)
            self.assertIsNotNone(artifacts.events_oasis_h5)
            self.assertIsNotNone(artifacts.classification_h5)
            self.assertIsNotNone(artifacts.average_projection_png)
            self.assertIsNotNone(artifacts.maximum_projection_png)

    def test_flat_v1_layout_still_resolves(self) -> None:
        """The pre-v2 flat layout resolves through the same glob."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for template in V2_LAYOUT.values():
                _touch(root / template.format(plane="VISp_0"))
            artifacts = file_handling.collect_plane_artifacts(root, "VISp_0")
            self.assertIsNotNone(artifacts.dff_h5)

    def test_optional_artifacts_warn_when_absent(self) -> None:
        """A skipped upstream stage warns instead of failing the run."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_v2_plane(root, "VISp_0", complete=False)
            with self.assertLogs(file_handling.logger, level=logging.WARNING):
                artifacts = file_handling.collect_plane_artifacts(root, "VISp_0")
            self.assertIsNone(artifacts.dff_h5)
            self.assertIsNone(artifacts.events_oasis_h5)
            self.assertIsNone(artifacts.classification_h5)

    def test_missing_required_artifact_names_pattern_and_directory(self) -> None:
        """The error is diagnosable: it names the glob and the directory."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError) as ctx:
                file_handling.collect_plane_artifacts(root, "VISp_0")
            message = str(ctx.exception)
            self.assertIn("VISp_0_*extraction.h5", message)
            self.assertIn(str(root), message)

    def test_unprefixed_fallback(self) -> None:
        """A single-plane session tolerates a differing upstream unique id."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "motion_correction" / "MOp2_3_0_extraction.h5")
            with self.assertLogs(file_handling.logger, level=logging.INFO):
                artifacts = file_handling.collect_plane_artifacts(
                    root, "plane_0", allow_unprefixed=True
                )
            self.assertEqual(artifacts.extraction_h5.name, "MOp2_3_0_extraction.h5")

    def test_unprefixed_fallback_still_fails_when_nothing_exists(self) -> None:
        """The fallback does not mask a genuinely missing artifact."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                file_handling.collect_plane_artifacts(Path(tmp), "plane_0", allow_unprefixed=True)

    def test_unprefixed_fallback_refuses_to_guess_between_planes(self) -> None:
        """An ambiguous mount must raise, not silently pick sorted()[0].

        The metadata says one plane but the mount holds four. Taking the first
        match published one plane's data under another plane's name and exited
        0, with only a WARNING as the signal — three planes silently absent
        and the traces timestamped from the nominal frame rate because the
        session routed as single-plane.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for plane in ("VISp_0", "VISl_1", "VISal_2", "VISam_3"):
                _touch(root / plane / "extraction" / f"{plane}_extraction.h5")
            with self.assertRaises(ValueError) as ctx:
                file_handling.collect_plane_artifacts(root, "plane_0", allow_unprefixed=True)
        message = str(ctx.exception)
        self.assertIn("plane_0", message)
        self.assertIn("4 files match", message)
        self.assertIn("VISal_2", message)


class TestCollectSessionArtifacts(unittest.TestCase):
    """The terminal fan-in: every plane resolved in one call."""

    def test_multiplane_preserves_plane_order(self) -> None:
        """Results are returned in the given plane order."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for plane in PLANES:
                _write_v2_plane(root, plane)
            artifacts = file_handling.collect_session_artifacts(root, PLANES)
            self.assertEqual([a.plane_id for a in artifacts], list(PLANES))
            self.assertEqual(
                artifacts[1].extraction_h5.name,
                "VISl_1_extraction.h5",
            )

    def test_single_plane_enables_the_unprefixed_fallback(self) -> None:
        """One plane id means the upstream unique id may differ."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "extraction" / "somethingelse_extraction.h5")
            artifacts = file_handling.collect_session_artifacts(root, ("plane_0",))
            self.assertEqual(len(artifacts), 1)

    def test_no_planes_raises(self) -> None:
        """An empty plane list is a metadata failure, not an empty NWB."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                file_handling.collect_session_artifacts(Path(tmp), ())
            self.assertIn("No imaging planes", str(ctx.exception))


class TestSideInputs(unittest.TestCase):
    """Epoch and sync file discovery."""

    def test_find_epoch_locations_nested(self) -> None:
        """Motion correction's epoch file is found at any depth."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = _touch(root / "plane_0" / "motion_correction" / "epoch_locations.json")
            self.assertEqual(file_handling.find_epoch_locations(root), expected)

    def test_find_epoch_locations_absent(self) -> None:
        """A session without epochs yields None."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(file_handling.find_epoch_locations(Path(tmp)))

    def test_find_sync_file_prefers_single_plane_data_main(self) -> None:
        """The single-plane pophys/data_main.h5 wins when present."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = _touch(root / "pophys" / "data_main.h5")
            _touch(root / "behavior" / "sess_sync.h5")
            self.assertEqual(file_handling.find_sync_file(root), expected)

    def test_find_sync_file_behavior_folder(self) -> None:
        """A behavior/*.h5 is used when there is no data_main.h5."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = _touch(root / "behavior" / "sess_sync.h5")
            self.assertEqual(file_handling.find_sync_file(root), expected)

    def test_find_sync_file_ophys_folder(self) -> None:
        """Any *ophys/*.h5 is the last resort."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = _touch(root / "mpophys" / "sess_sync.h5")
            self.assertEqual(file_handling.find_sync_file(root), expected)

    def test_find_sync_file_absent_logs_and_returns_none(self) -> None:
        """A missing sync file is logged with every pattern tried."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs(file_handling.logger, level=logging.ERROR) as logs:
                self.assertIsNone(file_handling.find_sync_file(Path(tmp)))
            self.assertIn("pophys/data_main.h5", logs.output[0])
            self.assertIn("behavior/*.h5", logs.output[0])
            self.assertIn(str(tmp), logs.output[0])

    def test_find_sync_file_directory_component_names_the_immediate_parent(self) -> None:
        """A pattern's directory component must match the file's own parent."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "behavior" / "nested" / "sess_sync.h5")
            _touch(root / "pophys_extra" / "data_main.h5")
            _touch(root / "other" / "data_main.h5")
            with self.assertLogs(file_handling.logger, level=logging.ERROR):
                self.assertIsNone(file_handling.find_sync_file(root))


class TestSymlinkedDirectories(unittest.TestCase):
    """Discovery through a mount whose entries are symlinks to directories elsewhere.

    Workflow engines stage task inputs this way, and ``Path.rglob`` does not
    descend into a symlinked directory.
    """

    def setUp(self) -> None:
        """Link ``mount/staged`` to a real directory outside the mount."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.real = Path(tmp.name) / "work" / "staged"
        self.real.mkdir(parents=True)
        self.root = Path(tmp.name) / "mount"
        self.root.mkdir()
        (self.root / "staged").symlink_to(self.real, target_is_directory=True)

    def _assert_sync_file_behind_link(self, relative: str) -> None:
        """Assert ``find_sync_file`` resolves ``relative`` staged behind the link."""
        _touch(self.real / relative)
        self.assertEqual(
            file_handling.find_sync_file(self.root),
            self.root / "staged" / relative,
        )

    def test_find_artifact_follows_the_link(self) -> None:
        """An artifact behind the link is found at its path under the mount."""
        _touch(self.real / "VISp_0" / "dff" / "VISp_0_dff.h5")
        self.assertEqual(
            file_handling.find_artifact(self.root, "dff.h5", "VISp_0"),
            self.root / "staged" / "VISp_0" / "dff" / "VISp_0_dff.h5",
        )

    def test_find_artifacts_follows_every_link_in_sorted_order(self) -> None:
        """One link per staged input; matches from all of them come back sorted."""
        other = self.real.parent / "other"
        _touch(other / "VISl_1_dff.h5")
        (self.root / "other").symlink_to(other, target_is_directory=True)
        _touch(self.real / "VISp_0_dff.h5")
        self.assertEqual(
            file_handling.find_artifacts(self.root, "dff.h5"),
            [
                self.root / "other" / "VISl_1_dff.h5",
                self.root / "staged" / "VISp_0_dff.h5",
            ],
        )

    def test_a_link_back_to_the_mount_does_not_recurse_forever(self) -> None:
        """A symlink cycle yields each file once instead of looping."""
        _touch(self.real / "VISp_0_dff.h5")
        (self.real / "loop").symlink_to(self.root, target_is_directory=True)
        self.assertEqual(
            file_handling.find_artifacts(self.root, "dff.h5"),
            [self.root / "staged" / "VISp_0_dff.h5"],
        )

    def test_collect_session_artifacts_follows_the_link(self) -> None:
        """Every plane resolves from a layout that sits wholly behind the link."""
        for plane in PLANES:
            _write_v2_plane(self.real, plane)
        artifacts = file_handling.collect_session_artifacts(self.root, PLANES)
        self.assertEqual([plane.plane_id for plane in artifacts], list(PLANES))
        for plane in artifacts:
            name = f"{plane.plane_id}_extraction.h5"
            self.assertEqual(
                plane.extraction_h5,
                self.root / "staged" / plane.plane_id / "extraction" / name,
            )
            self.assertTrue(all(value is not None for value in vars(plane).values()))

    def test_find_epoch_locations_follows_the_link(self) -> None:
        """The epoch file behind the link is found."""
        _touch(self.real / "plane_0" / "motion_correction" / "epoch_locations.json")
        self.assertEqual(
            file_handling.find_epoch_locations(self.root),
            self.root / "staged" / "plane_0" / "motion_correction" / "epoch_locations.json",
        )

    def test_find_sync_file_data_main_follows_the_link(self) -> None:
        """The single-plane pophys/data_main.h5 is found behind the link."""
        self._assert_sync_file_behind_link("pophys/data_main.h5")

    def test_find_sync_file_behavior_folder_follows_the_link(self) -> None:
        """A behavior/*.h5 is found behind the link."""
        self._assert_sync_file_behind_link("behavior/sess_sync.h5")

    def test_find_sync_file_ophys_folder_follows_the_link(self) -> None:
        """A *ophys/*.h5 is found behind the link."""
        self._assert_sync_file_behind_link("mpophys/sess_sync.h5")


class TestArtifactContract(unittest.TestCase):
    """The artifact table itself is part of the cross-step contract."""

    def test_only_consumed_artifacts_are_globbed(self) -> None:
        """The v1 dict's unconsumed entries were dropped, not carried."""
        self.assertEqual(
            set(file_handling.ARTIFACT_PARTS),
            {
                "extraction_h5",
                "dff_h5",
                "events_oasis_h5",
                "classification_h5",
                "average_projection_png",
                "maximum_projection_png",
            },
        )

    def test_extraction_is_the_only_required_artifact(self) -> None:
        """Traces and masks are mandatory; everything else degrades."""
        self.assertEqual(file_handling.REQUIRED_ARTIFACTS, ("extraction_h5",))


if __name__ == "__main__":
    unittest.main()
