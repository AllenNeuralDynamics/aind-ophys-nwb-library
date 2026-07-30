"""Path helpers for provenance recorded in the emitted metadata.

Kept import-light (stdlib only) so it is cheap to import and testable without
pulling in pynwb/numpy via ``job``.
"""

from pathlib import Path


def relative_to_root(root: Path, path: Path) -> str:
    """Return ``path`` relative to ``root``, as a string.

    Both sides are resolved first. Inside the pipeline ``/results`` and
    ``/data`` are symlinks into the task work directory, so comparing a
    resolved root against an unresolved path raises ``ValueError`` and leaks
    an absolute container path (e.g. ``/results/nwb``) into the metadata.

    Parameters
    ----------
    root : pathlib.Path
        Directory the result should be relative to (the results or data root).
    path : pathlib.Path
        Path to express relative to ``root``.

    Returns
    -------
    str
        ``path`` relative to ``root``, or the absolute ``path`` when it is
        genuinely outside ``root`` (so provenance is never dropped entirely).
    """
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)
