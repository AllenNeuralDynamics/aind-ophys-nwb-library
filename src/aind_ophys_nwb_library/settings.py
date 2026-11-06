"""CLI/environment settings for NWB packaging."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

DEFAULT_EXTENSION_NAMESPACE = Path("ndx-aibs-behavior-ophys.namespace.yaml")

#: Settings field recorded by basename only.
EPHEMERAL_PATH_FIELD = "extension_namespace"


class NwbSettings(BaseSettings, cli_parse_args=True):
    """Settings for NWB packaging using pydantic-settings."""

    input_dir: Path = Field(
        default=Path("/data"),
        description="Directory holding the 'raw' and 'processed' input subdirectories.",
    )
    output_dir: Path = Field(
        default=Path("/results"),
        description="Directory where the NWB Zarr store and metadata are written.",
    )
    raw_subdir: str = Field(
        default="raw",
        description="Subdirectory of input_dir holding the raw asset.",
    )
    processed_subdir: str = Field(
        default="processed",
        description="Subdirectory of input_dir holding the upstream step results.",
    )
    nwb_name: str = Field(
        default="pophys.nwb.zarr",
        description="Basename of the NWB Zarr store, written at the root of output_dir.",
    )
    extension_namespace: Path = Field(
        default=DEFAULT_EXTENSION_NAMESPACE,
        description=(
            "ndx-aibs-behavior-ophys namespace YAML defining the per-plane lab "
            "metadata; a relative path resolves against the working directory."
        ),
    )
    skip_lab_metadata: bool = Field(
        default=False,
        description="Skip the ndx-aibs-behavior-ophys per-plane lab metadata.",
    )

    def recorded_parameters(self) -> dict:
        """Return the settings as recorded in ``Code.parameters``.

        ``extension_namespace`` is reduced to its basename.

        Returns
        -------
        dict
            JSON-serializable settings for the code block.
        """
        parameters = self.model_dump(mode="json")
        parameters[EPHEMERAL_PATH_FIELD] = Path(getattr(self, EPHEMERAL_PATH_FIELD)).name
        return parameters
