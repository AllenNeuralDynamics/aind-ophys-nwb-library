"""Tests for the v1/v2 metadata read layer and processing.json builders."""

import json
import os
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from aind_data_schema.core.processing import Processing
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import fields as shared_fields
from aind_pophys_metadata.runner import stage_guard

from aind_ophys_nwb_library.utils import metadata_utils as mu
from tests import fixtures


def _write(directory: Path, name: str, payload: dict) -> None:
    """Write one JSON metadata file into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload))


def _raw_mount(tmp: str, core_name: str, core: dict, **extra) -> Path:
    """Materialise a raw mount holding the core file plus optional siblings."""
    root = Path(tmp) / "raw"
    _write(root, core_name, core)
    for name, payload in extra.items():
        _write(root, f"{name}.json", payload)
    return root


class TestCoercion(unittest.TestCase):
    """Raw metadata values arrive as strings, ints or nothing."""

    def test_float_coercion(self):
        """Numeric strings coerce; None stays None."""
        self.assertEqual(mu._as_float("1.5"), 1.5)
        self.assertEqual(mu._as_float(2), 2.0)
        self.assertIsNone(mu._as_float(None))

    def test_float_coercion_warns_on_garbage(self):
        """A non-numeric value is logged rather than crashing the read."""
        with self.assertLogs(mu.logger, level="WARNING"):
            self.assertIsNone(mu._as_float("not-a-number"))

    def test_int_coercion_comes_from_the_shared_layer(self):
        """The shared coercion tolerates an explicit JSON null."""
        self.assertEqual(mu._as_int("3"), 3)
        self.assertEqual(mu._as_int(4.0), 4)
        self.assertIsNone(mu._as_int(None))
        self.assertIs(mu._as_int, shared_fields._as_int)


class TestPlaneMetadata(unittest.TestCase):
    """The frozen per-plane record."""

    def test_is_hashable(self):
        """A frozen dataclass with immutable fields must stay hashable."""
        plane = mu.PlaneMetadata(plane_id="VISp_0", plane_index=0)
        self.assertIsInstance(hash(plane), int)

    def test_location_string(self):
        """The NWB location string names structure and depth."""
        plane = mu.PlaneMetadata(plane_id="VISp_0", plane_index=0, structure="VISp", depth=150.0)
        self.assertEqual(plane.location, "Structure: VISp Depth: 150.0")

    def test_location_string_tolerates_missing_fields(self):
        """A partial-metadata plane never renders 'None'."""
        plane = mu.PlaneMetadata(plane_id="plane_0", plane_index=0)
        self.assertEqual(plane.location, "Structure: unknown Depth: unknown")


class TestPlaneRecordsV1(unittest.TestCase):
    """v1 ``ophys_fovs`` traversal."""

    def test_reads_every_fov_field(self):
        """All the fields the NWB builders consume are populated."""
        records = mu.plane_records_v1(fixtures.v1_session())
        self.assertEqual(len(records), 2)
        first = records[0]
        self.assertEqual(first.plane_index, 0)
        self.assertEqual(first.structure, "VISp")
        self.assertEqual(first.depth, 150.0)
        self.assertEqual(first.coupled_plane_index, 0)
        self.assertEqual(first.scale_factor, fixtures.SCALE_UM_PER_PX)
        self.assertEqual(first.width, fixtures.FOV_WIDTH)
        self.assertEqual(first.height, fixtures.FOV_HEIGHT)

    def test_bare_string_targeted_structure(self):
        """Early-v1 wrote a bare acronym string, not a dict."""
        session = fixtures.v1_session()
        session["data_streams"][0]["ophys_fovs"][0]["targeted_structure"] = "DLS"
        self.assertEqual(mu.plane_records_v1(session)[0].structure, "DLS")

    def test_fov_without_index_defaults_to_zero(self):
        """A single-plane FOV may omit index entirely."""
        session = fixtures.v1_session(plane_count=1)
        del session["data_streams"][0]["ophys_fovs"][0]["index"]
        self.assertEqual(mu.plane_records_v1(session)[0].plane_index, 0)

    def test_scale_factor_unit_fallback(self):
        """fov_scale_factor_unit is used when fov_coordinate_unit is absent."""
        session = fixtures.v1_session(plane_count=1)
        fov = session["data_streams"][0]["ophys_fovs"][0]
        del fov["fov_coordinate_unit"]
        fov["fov_scale_factor_unit"] = "millimeter"
        self.assertEqual(mu.plane_records_v1(session)[0].coordinate_unit, "millimeter")

    def test_default_units_when_all_absent(self):
        """Missing units fall back to micrometer, not None."""
        session = {"data_streams": [{"ophys_fovs": [{}]}]}
        record = mu.plane_records_v1(session)[0]
        self.assertEqual(record.coordinate_unit, mu.DEFAULT_LENGTH_UNIT)
        self.assertEqual(record.depth_unit, mu.DEFAULT_LENGTH_UNIT)

    def test_no_streams(self):
        """A session with no data_streams yields no records."""
        self.assertEqual(mu.plane_records_v1({}), [])


class TestPlaneRecordsV2(unittest.TestCase):
    """v2 ``configurations -> images -> planes`` traversal."""

    def test_reads_every_v2_field(self):
        """The v1 FOV fields all resolve from their v2 locations."""
        records = mu.plane_records_v2(fixtures.v2_acquisition())
        self.assertEqual(len(records), 2)
        first = records[0]
        self.assertEqual(first.plane_index, 0)
        self.assertEqual(first.structure, "VISp")
        self.assertEqual(first.depth, 150.0)
        self.assertEqual(first.coupled_plane_index, 0)
        self.assertEqual(first.scale_factor, fixtures.SCALE_UM_PER_PX)
        self.assertEqual(first.coordinate_unit, "micrometer")
        self.assertEqual(first.width, fixtures.FOV_WIDTH)
        self.assertEqual(first.height, fixtures.FOV_HEIGHT)

    def test_non_imaging_configuration_is_skipped(self):
        """A stream carrying a non-imaging config contributes no planes."""
        acquisition = fixtures.v2_acquisition()
        acquisition["data_streams"].append({"configurations": [{"object_type": "Laser config"}]})
        self.assertEqual(len(mu.plane_records_v2(acquisition)), 2)

    def test_non_planar_image_is_skipped(self):
        """An ImageSPIM under the same config is ignored."""
        acquisition = fixtures.v2_acquisition()
        config = acquisition["data_streams"][0]["configurations"][0]
        config["images"].append({"object_type": "Image SPIM", "planes": [{}]})
        self.assertEqual(len(mu.plane_records_v2(acquisition)), 2)

    def test_missing_dimensions_and_transform(self):
        """Absent dimensions/transform leave width/height/scale as None."""
        acquisition = fixtures.v2_acquisition(plane_count=1)
        image = acquisition["data_streams"][0]["configurations"][0]["images"][0]
        del image["dimensions"]
        image["image_to_acquisition_transform"] = []
        record = mu.plane_records_v2(acquisition)[0]
        self.assertIsNone(record.width)
        self.assertIsNone(record.height)
        self.assertIsNone(record.scale_factor)

    def test_one_dimensional_scale(self):
        """A single-axis dimensions Scale sets width but not height."""
        acquisition = fixtures.v2_acquisition(plane_count=1)
        image = acquisition["data_streams"][0]["configurations"][0]["images"][0]
        image["dimensions"]["scale"] = [256]
        record = mu.plane_records_v2(acquisition)[0]
        self.assertEqual(record.width, 256)
        self.assertIsNone(record.height)

    def test_first_scale_skips_non_scale_transforms(self):
        """A Translation ahead of the Scale does not shadow it."""
        transforms = [
            {"object_type": "Translation", "translation": [1.0, 2.0]},
            "not-a-dict",
            {"object_type": "Scale", "scale": []},
            {"object_type": "Scale", "scale": [0.5]},
        ]
        self.assertEqual(mu._first_scale(transforms), 0.5)

    def test_first_scale_with_no_transforms(self):
        """No transform list yields None."""
        self.assertIsNone(mu._first_scale(None))


class TestAttachPlaneIds(unittest.TestCase):
    """Positional pairing to the shared layer's canonical plane ids."""

    def test_sorts_by_plane_index(self):
        """Records are ordered by plane index before pairing."""
        records = [
            mu.PlaneMetadata(plane_id="", plane_index=1, structure="VISl"),
            mu.PlaneMetadata(plane_id="", plane_index=0, structure="VISp"),
        ]
        planes = mu.attach_plane_ids(records, ("VISp_0", "VISl_1"))
        self.assertEqual([p.plane_id for p in planes], ["VISp_0", "VISl_1"])
        self.assertEqual([p.structure for p in planes], ["VISp", "VISl"])

    def test_preserves_every_rich_field(self):
        """Nothing is dropped when the id is attached."""
        record = mu.PlaneMetadata(
            plane_id="",
            plane_index=0,
            structure="VISp",
            depth=150.0,
            depth_unit="micrometer",
            coupled_plane_index=0,
            scale_factor=0.78,
            coordinate_unit="micrometer",
            width=512,
            height=512,
        )
        (plane,) = mu.attach_plane_ids([record], ("VISp_0",))
        self.assertEqual(plane.scale_factor, 0.78)
        self.assertEqual(plane.coupled_plane_index, 0)
        self.assertEqual(plane.width, 512)
        self.assertEqual(plane.height, 512)
        self.assertEqual(plane.depth_unit, "micrometer")
        self.assertEqual(plane.coordinate_unit, "micrometer")

    def test_length_mismatch_raises(self):
        """A traversal disagreement is loud, never silently truncated."""
        records = [mu.PlaneMetadata(plane_id="", plane_index=0)]
        with self.assertRaises(ValueError) as ctx:
            mu.attach_plane_ids(records, ("VISp_0", "VISl_1"))
        self.assertIn("disagree", str(ctx.exception))


class TestWavelengths(unittest.TestCase):
    """Excitation and emission wavelength reads."""

    def test_v1_excitation(self):
        """v1 reads data_streams[*].light_sources[0].wavelength."""
        self.assertEqual(
            mu.get_excitation_wavelength("v1", fixtures.v1_session()),
            float(fixtures.EXCITATION_NM),
        )

    def test_v1_excitation_absent(self):
        """No light source yields None, not NaN."""
        session = fixtures.v1_session()
        del session["data_streams"][0]["light_sources"]
        self.assertIsNone(mu.get_excitation_wavelength("v1", session))

    def test_v2_excitation(self):
        """v2 reads the LaserConfig under ImagingConfig.channels."""
        self.assertEqual(
            mu.get_excitation_wavelength("v2", fixtures.v2_acquisition()),
            float(fixtures.EXCITATION_NM),
        )

    def test_v2_excitation_skips_non_laser_sources(self):
        """An LED config is not read as an excitation laser."""
        acquisition = fixtures.v2_acquisition()
        channel = acquisition["data_streams"][0]["configurations"][0]["channels"][0]
        channel["light_sources"] = [
            {"object_type": "Light emitting diode config", "wavelength": 470}
        ]
        self.assertIsNone(mu.get_excitation_wavelength("v2", acquisition))

    def test_v2_emission(self):
        """The v2 channel's emission wavelength replaces the hardcoded 514."""
        self.assertEqual(
            mu.get_emission_wavelength("v2", fixtures.v2_acquisition()),
            float(fixtures.EMISSION_NM),
        )

    def test_v2_emission_absent(self):
        """A channel without an emission wavelength yields None."""
        acquisition = fixtures.v2_acquisition()
        channel = acquisition["data_streams"][0]["configurations"][0]["channels"][0]
        del channel["emission_wavelength"]
        self.assertIsNone(mu.get_emission_wavelength("v2", acquisition))

    def test_v1_has_no_emission(self):
        """v1 had no per-channel emission wavelength at all."""
        self.assertIsNone(mu.get_emission_wavelength("v1", fixtures.v1_session()))


class TestInjectionMaterial(unittest.TestCase):
    """The indicator read, which must span the v1 and v2 procedure shapes."""

    def test_v1_shape(self):
        """The v1 subject_procedures nesting resolves."""
        self.assertEqual(
            mu.get_injection_material(fixtures.procedures()),
            fixtures.INJECTION_MATERIAL,
        )

    def test_deeper_nesting(self):
        """A differently-nested (v2) shape resolves through the same search."""
        payload = {"components": [{"steps": [{"injection_materials": [{"name": "GCaMP8s"}]}]}]}
        self.assertEqual(mu.get_injection_material(payload), "GCaMP8s")

    def test_missing_returns_none(self):
        """No injection material anywhere yields None."""
        self.assertIsNone(mu.get_injection_material({"subject_procedures": []}))

    def test_material_without_name(self):
        """An injection material with no name is skipped, not crashed on."""
        self.assertIsNone(mu.get_injection_material({"injection_materials": [{}]}))

    def test_scalar_leaves_are_walked_over(self):
        """Free-text and numeric leaves are traversed without raising."""
        payload = {"notes": "no injections", "count": 3, "steps": [None, 1.5]}
        self.assertIsNone(mu.get_injection_material(payload))

    def test_absent_file(self):
        """A missing procedures.json yields None."""
        self.assertIsNone(mu.get_injection_material(None))


class TestSessionType(unittest.TestCase):
    """Routing is by plane count, not by the v1-only platform field."""

    def test_multiplane(self):
        """More than one plane is multiplane."""
        self.assertIs(mu.session_type(4, False), mu.SessionType.MULTIPLANE)

    def test_single_plane(self):
        """One plane with no epoch file is plain single-plane."""
        self.assertIs(mu.session_type(1, False), mu.SessionType.SINGLE_PLANE)

    def test_single_plane_with_epochs(self):
        """One plane plus epoch_locations.json is the epoch variant."""
        self.assertIs(
            mu.session_type(1, True),
            mu.SessionType.SINGLE_PLANE_WITH_EPOCHS,
        )


class TestLoadNwbMetadata(unittest.TestCase):
    """The fail-fast read at the top of run()."""

    def test_v1_input(self):
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

    def test_v2_input(self):
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

    def test_single_plane_gets_plane_zero_id(self):
        """One plane uses the canonical plane_0 id, never {structure}_0."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(tmp, "acquisition.json", fixtures.v2_acquisition(plane_count=1))
            metadata = mu.load_nwb_metadata(root)
        self.assertEqual(metadata.plane_ids, ("plane_0",))
        self.assertTrue(metadata.single_plane)

    def test_platform_json_frame_rate_fallback(self):
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

    def test_missing_frame_rate_raises(self):
        """No frame rate anywhere is a hard failure naming the file."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(tmp, "session.json", fixtures.v1_session_without_frame_rate())
            with self.assertRaises(ValueError) as ctx:
                mu.load_nwb_metadata(root)
        self.assertIn("session.json", str(ctx.exception))
        self.assertIn("frame rate", str(ctx.exception))

    def test_no_planes_raises(self):
        """Zero planes is a hard failure, not an empty NWB."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(tmp, "session.json", fixtures.v1_session_without_planes())
            with self.assertRaises(ValueError) as ctx:
                mu.load_nwb_metadata(root)
        self.assertIn("No imaging planes", str(ctx.exception))

    def test_missing_core_file_raises(self):
        """Neither session.json nor acquisition.json is a FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mu.load_nwb_metadata(Path(tmp))

    def test_both_core_files_is_ambiguous(self):
        """A mount carrying both versions refuses to guess."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _raw_mount(tmp, "session.json", fixtures.v1_session())
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            with self.assertRaises(ValueError):
                mu.load_nwb_metadata(root)

    def test_subject_id_only_in_subject_json(self):
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

    def test_url_is_the_library_landing_page(self):
        """No /tree/main/code suffix and no branch ref."""
        code = mu.build_nwb_code({"input_dir": "/data"})
        self.assertEqual(
            code.url,
            f"https://github.com/AllenNeuralDynamics/{mu.LIBRARY_NAME}",
        )

    def test_version_is_non_empty(self):
        """An empty version leaves the step with no code identity at all."""
        self.assertTrue(mu.build_nwb_code({}).version)

    def test_version_ignores_the_capsule_environment(self):
        """The backing library is authoritative, not the wrapper's VERSION."""
        with patch.dict(os.environ, {"VERSION": "1.2.3"}, clear=True):
            code = mu.build_nwb_code({})
        self.assertNotEqual(code.version, "1.2.3")
        self.assertEqual(code.version, version(mu.LIBRARY_NAME))

    def test_parameters_round_trip_through_generic_model(self):
        """Code.parameters becomes a _GenericModel, read back via model_dump."""
        code = mu.build_nwb_code({"num_planes": 2})
        self.assertEqual(code.parameters.model_dump()["num_planes"], 2)

    def test_ephemeral_parameter_path_is_rejected(self):
        """A Nextflow task scratch path must never reach permanent metadata."""
        with self.assertRaises(ValueError):
            mu.build_nwb_code({"extension_namespace": "/tmp/nxf.abc/capsule/code/ext.yaml"})


class TestAttachInputData(unittest.TestCase):
    """Input provenance, resolved after the code block already exists."""

    def test_names_are_wrapped_as_data_assets(self):
        """Each resolved artifact becomes one DataAsset name."""
        code = mu.build_nwb_code({})
        mu.attach_input_data(code, ["VISp_0/extraction/VISp_0_extraction.h5"])
        self.assertEqual(
            [asset.name for asset in code.input_data],
            ["VISp_0/extraction/VISp_0_extraction.h5"],
        )

    def test_ephemeral_path_is_rejected(self):
        """The guard build_code applies is not bypassed by attaching later."""
        code = mu.build_nwb_code({})
        with self.assertRaises(ValueError):
            mu.attach_input_data(code, ["/tmp/nxf.abc/capsule/results/x.h5"])


class TestStageDocument(unittest.TestCase):
    """The processing.json the shared stage guard writes for this capsule."""

    def _run_stage(self, output_dir: Path, upstream_names=None) -> Processing:
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

    def test_document_round_trips(self):
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

    def test_name_does_not_collide_with_the_converter(self):
        """Both capsules use FILE_FORMAT_CONVERSION and run once per session."""
        self.assertNotEqual(mu.DATA_PROCESS_NAME, ProcessName.FILE_FORMAT_CONVERSION.value)

    def test_upstream_names_become_dependency_graph_edges(self):
        """Upstream processes are flat siblings under this process's name."""
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                processing = self._run_stage(Path(tmp), ["VISp_0: dF/F estimation"])
        self.assertEqual(
            processing.dependency_graph,
            {mu.DATA_PROCESS_NAME: ["VISp_0: dF/F estimation"]},
        )

    def test_document_is_written_even_when_the_stage_raises(self):
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
