"""Tests for NwbSettings."""

import sys
import typing
import unittest
from pathlib import Path
from typing import Any, Optional, Sequence
from unittest.mock import patch

from aind_ophys_nwb_library.settings import NwbSettings


def _make_settings(argv: Optional[Sequence[str]] = None, **kwargs: Any) -> NwbSettings:
    """Construct NwbSettings with a controlled sys.argv."""
    with patch.object(sys, "argv", ["run_capsule.py"] + list(argv or [])):
        return NwbSettings(**kwargs)


def _sample_cli_value(annotation: Any) -> str:
    """Return a valid ``--name=value`` payload for a field annotation."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        annotation = args[0]
    if annotation is bool:
        return "true"
    if annotation is int:
        return "2"
    if annotation is float:
        return "1.5"
    if annotation is Path:
        return "/tmp/nwb-test"
    return "sample"


class TestNwbSettingsDefaults(unittest.TestCase):
    """Default field values match the Code Ocean mounts."""

    def test_defaults(self) -> None:
        """Defaults point at the standard Code Ocean mounts."""
        settings = _make_settings()
        self.assertEqual(settings.input_dir, Path("/data"))
        self.assertEqual(settings.output_dir, Path("/results"))
        self.assertEqual(settings.raw_subdir, "raw")
        self.assertEqual(settings.processed_subdir, "processed")
        self.assertEqual(settings.nwb_name, "pophys.nwb.zarr")
        self.assertFalse(settings.skip_lab_metadata)

    def test_extension_namespace_default_is_code_relative(self) -> None:
        """The extension YAML is resolved from the Code Ocean capsule's code/ folder."""
        settings = _make_settings()
        self.assertEqual(
            settings.extension_namespace,
            Path("ndx-aibs-behavior-ophys.namespace.yaml"),
        )
        self.assertFalse(settings.extension_namespace.is_absolute())

    def test_input_and_output_are_renamed_not_aliased(self) -> None:
        """Only ``input_dir``/``output_dir`` exist, with no ``*_directory`` aliases."""
        self.assertIn("input_dir", NwbSettings.model_fields)
        self.assertIn("output_dir", NwbSettings.model_fields)
        self.assertNotIn("input_directory", NwbSettings.model_fields)
        self.assertNotIn("output_directory", NwbSettings.model_fields)

    def test_bool_string_parsing(self) -> None:
        """Bools parse from strings such as ``true``."""
        settings = _make_settings(["--skip_lab_metadata=true"])
        self.assertTrue(settings.skip_lab_metadata)


class TestRecordedParameters(unittest.TestCase):
    """What reaches Code.parameters must outlive the task that ran it."""

    def test_ephemeral_namespace_path_is_reduced_to_a_basename(self) -> None:
        """A Nextflow task scratch path means nothing once the task ends."""
        settings = _make_settings(
            ["--extension_namespace=/tmp/nxf.abc123/capsule/code/ndx.namespace.yaml"]
        )
        self.assertEqual(
            settings.recorded_parameters()["extension_namespace"],
            "ndx.namespace.yaml",
        )

    def test_every_other_field_is_recorded_unchanged(self) -> None:
        """Only the one path field is rewritten."""
        settings = _make_settings()
        recorded = settings.recorded_parameters()
        self.assertEqual(recorded["input_dir"], "/data")
        self.assertEqual(recorded["nwb_name"], "pophys.nwb.zarr")
        self.assertEqual(set(recorded), set(NwbSettings.model_fields))


class TestSampleCliValue(unittest.TestCase):
    """The helper the app-panel contract test drives every field with."""

    def test_covers_every_annotation_this_settings_class_uses(self) -> None:
        """Each supported annotation yields a parseable payload."""
        self.assertEqual(_sample_cli_value(bool), "true")
        self.assertEqual(_sample_cli_value(int), "2")
        self.assertEqual(_sample_cli_value(float), "1.5")
        self.assertTrue(_sample_cli_value(Path).startswith("/"))
        self.assertEqual(_sample_cli_value(str), "sample")

    def test_optional_unwraps_to_the_inner_type(self) -> None:
        """Optional[float] is driven as a float, not as a string."""
        self.assertEqual(_sample_cli_value(typing.Optional[float]), "1.5")


class TestAppPanelContract(unittest.TestCase):
    """Every field must be drivable by a Code Ocean app panel."""

    def test_no_single_character_field_names(self) -> None:
        """No field name is one character long."""
        offenders = [name for name in NwbSettings.model_fields if len(name) == 1]
        self.assertEqual(offenders, [])

    def test_every_field_parses_as_named_argument(self) -> None:
        """Each field accepts the ``--name=value`` form the panel emits."""
        for name, field in NwbSettings.model_fields.items():
            with self.subTest(field=name):
                value = _sample_cli_value(field.annotation)
                settings = _make_settings([f"--{name}={value}"])
                self.assertIsNotNone(getattr(settings, name))

    def test_field_names_use_underscores(self) -> None:
        """pydantic-settings style: no hyphens in field names."""
        offenders = [name for name in NwbSettings.model_fields if "-" in name]
        self.assertEqual(offenders, [])

    def test_every_field_has_a_description(self) -> None:
        """A panel with no descriptions is unusable by an operator."""
        missing = [
            name for name, field in NwbSettings.model_fields.items() if not field.description
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
