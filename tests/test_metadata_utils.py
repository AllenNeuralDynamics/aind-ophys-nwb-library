"""Tests for the v1/v2 metadata read layer and processing.json builders."""

import json
import os
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from typing import Optional, Sequence
from unittest.mock import patch

import numpy as np
from aind_data_schema.core.processing import Processing
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import SchemaVersion
from aind_pophys_metadata.runner import stage_guard

from aind_ophys_nwb_library import nwb_builders as nb
from aind_ophys_nwb_library.utils import metadata_utils as mu
from tests import fixtures

LIBRARY = "aind-ophys-nwb-library"


def _write(directory: Path, name: str, payload: dict) -> None:
    """Write one JSON metadata file into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload))


def _raw_mount(tmp: str, core_name: str, core: dict, **extra: dict) -> Path:
    """Materialise a raw mount holding the core file plus optional siblings."""
    root = Path(tmp) / "raw"
    _write(root, core_name, core)
    for name, payload in extra.items():
        _write(root, f"{name}.json", payload)
    return root


class TestPlaneLocation(unittest.TestCase):
    """The one NWB-flavoured read of a shared plane record."""

    def test_location_string(self) -> None:
        """The NWB location string names structure and depth."""
        plane = {"plane_index": 0, "structure": "VISp", "depth": 150.0}
        self.assertEqual(mu.plane_location(plane), "Structure: VISp Depth: 150.0")

    def test_location_string_tolerates_missing_fields(self) -> None:
        """A partial-metadata plane never renders 'None'."""
        self.assertEqual(
            mu.plane_location({"plane_index": 0}),
            "Structure: unknown Depth: unknown",
        )


class TestPlaneFields(unittest.TestCase):
    """Every per-plane field the NWB builders consume, read from any version.

    The shared fields must arrive populated, identically, for each input
    the step can be handed.
    """

    def _planes(self, core_name: str, core: dict) -> tuple:
        """Load one raw mount and return its identified plane records."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(tmp, core_name, core)
            return mu.load_nwb_metadata(root).planes

    def _assert_rich_fields(self, planes: tuple) -> None:
        """Assert the shared fields the NWB builders read off plane zero."""
        self.assertEqual(len(planes), 2)
        first = planes[0]
        self.assertEqual(first["plane_index"], 0)
        self.assertEqual(first["structure"], "VISp")
        self.assertEqual(first["depth"], 150.0)
        self.assertEqual(first["depth_unit"], "micrometer")
        self.assertEqual([plane[mu.PLANE_GROUP_KEY] for plane in planes], [0, 0])
        self.assertAlmostEqual(first["um_per_pixel"], fixtures.SCALE_UM_PER_PX)
        self.assertEqual(first["width"], fixtures.FOV_WIDTH)
        self.assertEqual(first["height"], fixtures.FOV_HEIGHT)

    def test_v1_input(self) -> None:
        """A v1 session yields fully populated plane records."""
        self._assert_rich_fields(self._planes("session.json", fixtures.v1_session()))

    def test_v2_input(self) -> None:
        """A v2 acquisition yields the same values from its own layout."""
        self._assert_rich_fields(self._planes("acquisition.json", fixtures.v2_acquisition()))

    def test_planes_are_ordered_and_named(self) -> None:
        """Plane ids come from the shared layer, ordered by plane index."""
        planes = self._planes("acquisition.json", fixtures.v2_acquisition())
        self.assertEqual([plane["plane_id"] for plane in planes], ["VISp_0", "VISl_1"])
        self.assertEqual([plane["plane_index"] for plane in planes], [0, 1])

    def test_v2_coupled_pairs_time_together(self) -> None:
        """Partners in a v2 acquisition share one temporal group's timestamps."""
        planes = self._planes("acquisition.json", fixtures.v2_acquisition(plane_count=4))
        edges = np.arange(20, dtype=float)
        per_plane = nb.sync_timestamps_per_plane(planes, edges)
        ids = [plane["plane_id"] for plane in planes]
        np.testing.assert_array_equal(per_plane[ids[0]], edges[0::2])
        np.testing.assert_array_equal(per_plane[ids[1]], edges[0::2])
        np.testing.assert_array_equal(per_plane[ids[2]], edges[1::2])
        np.testing.assert_array_equal(per_plane[ids[3]], edges[1::2])

    def test_planes_are_sorted_before_being_paired_with_ids(self) -> None:
        """Records arrive unsorted, so the id pairing must sort first."""
        session = fixtures.v1_session()
        session["data_streams"][0]["ophys_fovs"].reverse()
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(tmp, "session.json", session)
            planes = mu.load_nwb_metadata(root).planes
        self.assertEqual([plane["plane_index"] for plane in planes], [0, 1])
        self.assertEqual([plane["plane_id"] for plane in planes], ["VISp_0", "VISl_1"])


class TestPlaneGroups(unittest.TestCase):
    """Coupled-pair indices from either way of recording the coupling."""

    @staticmethod
    def _records(coupled: list) -> list:
        """Return plane records, in plane-index order, with these coupled indices."""
        return [
            {"plane_index": index, "coupled_plane_index": value}
            for index, value in enumerate(coupled)
        ]

    def test_v1_and_minimal_record_the_pair_index(self) -> None:
        """v1 and minimal documents already carry the pair index."""
        for schema in (SchemaVersion.V1, SchemaVersion.MINIMAL):
            with self.subTest(schema=schema):
                records = self._records([0, 0, 1, 1])
                self.assertEqual(mu.plane_groups(records, schema), [0, 0, 1, 1])

    def test_v2_partners_are_numbered_in_plane_order(self) -> None:
        """v2 names each plane's partner, so [1, 0, 3, 2] is pairs 0 and 1."""
        records = self._records([1, 0, 3, 2, 5, 4, 7, 6])
        self.assertEqual(
            mu.plane_groups(records, SchemaVersion.V2), [0, 0, 1, 1, 2, 2, 3, 3]
        )

    def test_v2_interleaved_partners_keep_their_pairing(self) -> None:
        """Partners need not be adjacent; the pairing, not the order, decides."""
        records = self._records([2, 3, 0, 1])
        self.assertEqual(mu.plane_groups(records, SchemaVersion.V2), [0, 1, 0, 1])

    def test_v2_uncoupled_planes_have_no_group(self) -> None:
        """A single-plane acquisition records no coupling."""
        records = self._records([None])
        self.assertEqual(mu.plane_groups(records, SchemaVersion.V2), [None])

    def test_v2_partner_that_does_not_name_the_plane_back_raises(self) -> None:
        """A one-sided coupling cannot be timed and fails the run."""
        for coupled in ([1, 1], [0, 0], [5, 0]):
            with self.subTest(coupled=coupled):
                with self.assertRaises(ValueError):
                    mu.plane_groups(self._records(coupled), SchemaVersion.V2)


class TestWavelengths(unittest.TestCase):
    """Excitation and emission wavelengths, read as normalized fields."""

    def _metadata(self, core_name: str, core: dict) -> mu.NwbMetadata:
        """Load one raw mount carrying only the named core file."""
        with tempfile.TemporaryDirectory() as tmp:
            return mu.load_nwb_metadata(_raw_mount(tmp, core_name, core))

    def test_v1_excitation(self) -> None:
        """A v1 light source supplies the excitation wavelength."""
        metadata = self._metadata("session.json", fixtures.v1_session())
        self.assertEqual(metadata.excitation_wavelength, float(fixtures.EXCITATION_NM))

    def test_v1_has_no_emission(self) -> None:
        """v1 carries no emission wavelength; the builder placeholder applies."""
        metadata = self._metadata("session.json", fixtures.v1_session())
        self.assertIsNone(metadata.emission_wavelength)

    def test_v2_wavelengths(self) -> None:
        """A v2 channel supplies both wavelengths."""
        metadata = self._metadata("acquisition.json", fixtures.v2_acquisition())
        self.assertEqual(metadata.excitation_wavelength, float(fixtures.EXCITATION_NM))
        self.assertEqual(metadata.emission_wavelength, float(fixtures.EMISSION_NM))

    def test_absent_excitation(self) -> None:
        """No light source yields None, not NaN."""
        session = fixtures.v1_session()
        del session["data_streams"][0]["light_sources"]
        self.assertIsNone(self._metadata("session.json", session).excitation_wavelength)


class TestInjectionMaterial(unittest.TestCase):
    """The indicator read, which must span the v1 and v2 procedure shapes."""

    def test_v1_shape(self) -> None:
        """The v1 subject_procedures nesting resolves."""
        self.assertEqual(
            mu.get_injection_material(fixtures.procedures()),
            fixtures.INJECTION_MATERIAL,
        )

    def test_deeper_nesting(self) -> None:
        """A differently-nested (v2) shape resolves through the same search."""
        payload = {"components": [{"steps": [{"injection_materials": [{"name": "GCaMP8s"}]}]}]}
        self.assertEqual(mu.get_injection_material(payload), "GCaMP8s")

    def test_missing_returns_none(self) -> None:
        """No injection material anywhere yields None."""
        self.assertIsNone(mu.get_injection_material({"subject_procedures": []}))

    def test_material_without_name(self) -> None:
        """An injection material with no name is skipped, not crashed on."""
        self.assertIsNone(mu.get_injection_material({"injection_materials": [{}]}))

    def test_scalar_leaves_are_walked_over(self) -> None:
        """Free-text and numeric leaves are traversed without raising."""
        payload = {"notes": "no injections", "count": 3, "steps": [None, 1.5]}
        self.assertIsNone(mu.get_injection_material(payload))

    def test_absent_file(self) -> None:
        """A missing procedures.json yields None."""
        self.assertIsNone(mu.get_injection_material(None))


class TestSessionType(unittest.TestCase):
    """Routing is by plane count, not by the v1-only platform field."""

    def test_multiplane(self) -> None:
        """More than one plane is multiplane."""
        self.assertIs(mu.session_type(4, False), mu.SessionType.MULTIPLANE)

    def test_single_plane(self) -> None:
        """One plane with no epoch file is plain single-plane."""
        self.assertIs(mu.session_type(1, False), mu.SessionType.SINGLE_PLANE)

    def test_single_plane_with_epochs(self) -> None:
        """One plane plus epoch_locations.json is the epoch variant."""
        self.assertIs(
            mu.session_type(1, True),
            mu.SessionType.SINGLE_PLANE_WITH_EPOCHS,
        )


class TestLoadNwbMetadata(unittest.TestCase):
    """The fail-fast read at the top of run()."""

    def test_v1_input(self) -> None:
        """A v1 raw mount reads end to end."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(
                tmp,
                "session.json",
                fixtures.v1_session(),
                subject=fixtures.subject(),
                procedures=fixtures.procedures(),
                data_description=fixtures.data_description(),
            )
            metadata = mu.load_nwb_metadata(root)
        self.assertEqual(metadata.schema_version, "v1")
        self.assertEqual(metadata.frame_rate, fixtures.FRAME_RATE)
        self.assertEqual(metadata.plane_ids, ("VISp_0", "VISl_1"))
        self.assertEqual(metadata.instrument_id, fixtures.INSTRUMENT_ID)
        self.assertEqual(metadata.subject_id, fixtures.SUBJECT_ID)
        self.assertEqual(metadata.dataset_name, fixtures.DATASET_NAME)
        self.assertEqual(metadata.indicator, fixtures.INJECTION_MATERIAL)
        self.assertEqual(metadata.genotype, fixtures.GENOTYPE)
        self.assertFalse(metadata.single_plane)

    def test_v2_input(self) -> None:
        """A v2 raw mount reads end to end and yields the same plane ids."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(
                tmp,
                "acquisition.json",
                fixtures.v2_acquisition(),
                subject=fixtures.subject(),
                data_description=fixtures.data_description(),
            )
            metadata = mu.load_nwb_metadata(root)
        self.assertEqual(metadata.schema_version, "v2")
        self.assertEqual(metadata.plane_ids, ("VISp_0", "VISl_1"))
        self.assertEqual(metadata.emission_wavelength, float(fixtures.EMISSION_NM))
        self.assertEqual(metadata.indicator, fixtures.GENOTYPE)

    def test_v2_subject_genotype_is_the_indicator(self) -> None:
        """v2 subject.json keeps the genotype under subject_details."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(
                tmp,
                "acquisition.json",
                fixtures.v2_acquisition(),
                subject={
                    "subject_id": fixtures.SUBJECT_ID,
                    "subject_details": {"genotype": fixtures.GENOTYPE},
                },
            )
            metadata = mu.load_nwb_metadata(root)
        self.assertEqual(metadata.genotype, fixtures.GENOTYPE)
        self.assertEqual(metadata.indicator, fixtures.GENOTYPE)

    def test_minimal_input(self) -> None:
        """A flat minimal metadata.json reads through the same normalized path.

        The read is version-agnostic, so a third document shape reaches the
        step as populated fields rather than being parsed as v2.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(tmp, "metadata.json", fixtures.minimal_metadata())
            metadata = mu.load_nwb_metadata(root)
        self.assertEqual(metadata.plane_ids, ("VISp_0", "VISl_1"))
        self.assertEqual(metadata.frame_rate, fixtures.FRAME_RATE)
        self.assertEqual(metadata.instrument_id, fixtures.INSTRUMENT_ID)
        self.assertEqual(metadata.subject_id, fixtures.SUBJECT_ID)
        self.assertEqual(metadata.emission_wavelength, float(fixtures.EMISSION_NM))
        self.assertEqual(metadata.planes[0]["scale_factor"], fixtures.SCALE_UM_PER_PX)

    def test_single_plane_gets_plane_zero_id(self) -> None:
        """One plane uses the canonical plane_0 id, never {structure}_0."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(tmp, "acquisition.json", fixtures.v2_acquisition(plane_count=1))
            metadata = mu.load_nwb_metadata(root)
        self.assertEqual(metadata.plane_ids, ("plane_0",))
        self.assertTrue(metadata.single_plane)

    def test_platform_json_frame_rate_fallback(self) -> None:
        """platform.json supplies the frame rate when the core file omits it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(
                tmp,
                "session.json",
                fixtures.v1_session_without_frame_rate(),
                platform=fixtures.platform(7.5),
            )
            metadata = mu.load_nwb_metadata(root)
        self.assertEqual(metadata.frame_rate, 7.5)

    def test_missing_frame_rate_raises(self) -> None:
        """No frame rate anywhere is a hard failure naming the file."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(tmp, "session.json", fixtures.v1_session_without_frame_rate())
            with self.assertRaises(ValueError) as ctx:
                mu.load_nwb_metadata(root)
        self.assertIn("session.json", str(ctx.exception))
        self.assertIn("frame rate", str(ctx.exception))

    def test_no_planes_raises(self) -> None:
        """Zero planes is a hard failure, not an empty NWB."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(tmp, "session.json", fixtures.v1_session_without_planes())
            with self.assertRaises(ValueError) as ctx:
                mu.load_nwb_metadata(root)
        self.assertIn("No imaging planes", str(ctx.exception))

    def test_missing_core_file_raises(self) -> None:
        """Neither session.json nor acquisition.json is a FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mu.load_nwb_metadata(Path(tmp))

    def test_both_core_files_is_ambiguous(self) -> None:
        """A mount carrying both versions refuses to guess."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(tmp, "session.json", fixtures.v1_session())
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            with self.assertRaises(ValueError):
                mu.load_nwb_metadata(root)

    def test_subject_id_only_in_subject_json(self) -> None:
        """subject.json is the fallback for a core file without subject_id."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(
                tmp,
                "session.json",
                fixtures.v1_session(subject_id=None),
                subject=fixtures.subject(),
            )
            metadata = mu.load_nwb_metadata(root)
        self.assertEqual(metadata.subject_id, fixtures.SUBJECT_ID)


class TestCodeBlock(unittest.TestCase):
    """The Code block identifying which artifact produced the NWB."""

    def test_identity_is_the_installed_library(self) -> None:
        """name, version and url all describe this library's package."""
        code = mu.build_nwb_code({"input_dir": "/data"})
        self.assertEqual(code.name, LIBRARY)
        self.assertEqual(code.version, version(LIBRARY))
        self.assertTrue(code.version)
        self.assertEqual(code.url, f"https://github.com/AllenNeuralDynamics/{LIBRARY}")

    def test_version_ignores_the_capsule_environment(self) -> None:
        """The backing library is authoritative, not the wrapper's VERSION."""
        with patch.dict(os.environ, {"VERSION": "1.2.3"}, clear=True):
            code = mu.build_nwb_code({})
        self.assertNotEqual(code.version, "1.2.3")
        self.assertEqual(code.version, version(LIBRARY))

    def test_parameters_round_trip_through_generic_model(self) -> None:
        """Code.parameters becomes a _GenericModel, read back via model_dump."""
        code = mu.build_nwb_code({"num_planes": 2})
        self.assertEqual(code.parameters.model_dump()["num_planes"], 2)


class TestAttachInputData(unittest.TestCase):
    """Input provenance, resolved after the code block already exists."""

    def test_names_are_wrapped_as_data_assets(self) -> None:
        """Each resolved artifact becomes one DataAsset name."""
        code = mu.build_nwb_code({})
        mu.attach_input_data(code, ["VISp_0/extraction/VISp_0_extraction.h5"])
        self.assertEqual(
            [asset.name for asset in code.input_data],
            ["VISp_0/extraction/VISp_0_extraction.h5"],
        )


class TestStageDocument(unittest.TestCase):
    """The processing.json the shared stage guard writes for this step."""

    def _run_stage(
        self,
        output_dir: Path,
        upstream_names: Optional[Sequence[str]] = None,
    ) -> Processing:
        """Drive one stage through the guard and read the document back."""
        code = mu.build_nwb_code({"input_dir": "/data"})
        with stage_guard(
            output_dir,
            ProcessName.FILE_FORMAT_CONVERSION,
            code,
            name=mu.DATA_PROCESS_NAME,
            output_path="nwb",
        ) as ctx:
            ctx.upstream_names = list(upstream_names or [])
            ctx.output_parameters["num_planes"] = 2
        return Processing.model_validate_json((output_dir / "processing.json").read_text())

    def test_document_round_trips(self) -> None:
        """The written file validates back into a core v2 Processing."""
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                processing = self._run_stage(Path(tmp))
        process = processing.data_processes[0]
        self.assertEqual(process.process_type, ProcessName.FILE_FORMAT_CONVERSION)
        self.assertEqual(process.name, mu.DATA_PROCESS_NAME)
        self.assertEqual(str(process.output_path), "nwb")
        self.assertEqual(process.experimenters, [])
        self.assertIsNotNone(process.resources)

    def test_name_does_not_collide_with_the_converter(self) -> None:
        """Both steps use FILE_FORMAT_CONVERSION and run once per session."""
        self.assertNotEqual(mu.DATA_PROCESS_NAME, ProcessName.FILE_FORMAT_CONVERSION.value)

    def test_upstream_names_become_dependency_graph_edges(self) -> None:
        """Upstream processes are flat siblings under this process's name."""
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                processing = self._run_stage(Path(tmp), ["VISp_0: dF/F estimation"])
        self.assertEqual(
            processing.dependency_graph,
            {mu.DATA_PROCESS_NAME: ["VISp_0: dF/F estimation"]},
        )

    def test_document_is_written_even_when_the_stage_raises(self) -> None:
        """A failed stage that leaves no metadata looks like one that never ran."""
        code = mu.build_nwb_code({})
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp)
                with self.assertRaises(RuntimeError):
                    with stage_guard(
                        output_dir,
                        ProcessName.FILE_FORMAT_CONVERSION,
                        code,
                        name=mu.DATA_PROCESS_NAME,
                    ):
                        raise RuntimeError("boom")
                processing = Processing.model_validate_json(
                    (output_dir / "processing.json").read_text()
                )
        self.assertIn("boom", processing.data_processes[0].notes)


if __name__ == "__main__":
    unittest.main()
