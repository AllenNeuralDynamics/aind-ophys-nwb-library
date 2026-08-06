"""Re-export of the shared provenance path helper.

The implementation lives in ``aind_pophys_metadata.paths`` so all nine
libraries share one copy. This module only keeps the import path stable.
"""

from aind_pophys_metadata.paths import relative_to_root

__all__ = ["relative_to_root"]
