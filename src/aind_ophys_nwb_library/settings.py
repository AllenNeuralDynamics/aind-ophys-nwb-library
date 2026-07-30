"""CLI/environment settings for NWB packaging of ophys pipeline outputs.

Kept import-light (pydantic + stdlib only) so it stays cheap to import, free
of pynwb / hdmf-zarr / aind-data-schema side effects, and discoverable by
``auto-app-panel`` without the heavy dependencies installed.

Every field name is at least two characters: pydantic-settings renders a
one-character field as a *short* option (``-i``), which a Code Ocean app panel
can never emit because it only produces ``--name=value``. The v1 capsule used
argparse with ``--input_directory`` / ``--output_directory``; those are renamed
here to ``input_dir`` / ``output_dir`` to match every other library in the
pipeline (which is also why the pipeline's ``ophys_nwb`` invocation must be
updated -- see the README).
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

DEFAULT_EXTENSION_NAMESPACE = Path("ndx-aibs-behavior-ophys.namespace.yaml")


class NwbSettings(BaseSettings, cli_parse_args=True):
    """Settings for NWB packaging using pydantic-settings."""

    input_dir: Path = Field(
        default=Path("/data"),
        description=(
            "Directory holding the mounted inputs. Expected to contain a "
            "'raw' subdirectory with the core acquisition metadata and a "
            "'processed' subdirectory with every upstream capsule's output."
        ),
    )
    output_dir: Path = Field(
        default=Path("/results"),
        description="Directory where the NWB Zarr store and metadata are written.",
    )
    raw_subdir: str = Field(
        default="raw",
        description=(
            "Name of the subdirectory of input_dir holding the raw asset "
            "(core metadata, sync file, pophys movies)."
        ),
    )
    processed_subdir: str = Field(
        default="processed",
        description=(
            "Name of the subdirectory of input_dir holding every upstream " "capsule's results."
        ),
    )
    nwb_name: str = Field(
        default="pophys.nwb.zarr",
        description=(
            "Basename of the NWB Zarr store written inside output_dir. It is "
            "written into a subdirectory, never at the results root, so that "
            "the pipeline's basename-flattening publishDir cannot collide."
        ),
    )
    extension_namespace: Path = Field(
        default=DEFAULT_EXTENSION_NAMESPACE,
        description=(
            "Path to the ndx-aibs-behavior-ophys namespace YAML that defines "
            "the per-plane OphysMetadataSchema lab metadata. Resolved "
            "relative to the working directory (the capsule's code/ folder) "
            "when not absolute."
        ),
    )
    skip_lab_metadata: bool = Field(
        default=False,
        description=(
            "Skip the ndx-aibs-behavior-ophys per-plane lab metadata. Use "
            "only when the extension YAML is unavailable; the resulting NWB "
            "omits imaging_depth / imaging_plane_group / field_of_view_*."
        ),
    )
    verify: bool = Field(
        default=False,
        description=(
            "Round-trip the emitted processing.json and quality_control.json "
            "through the core v2 schema objects, and re-open the NWB Zarr "
            "store, after writing."
        ),
    )
