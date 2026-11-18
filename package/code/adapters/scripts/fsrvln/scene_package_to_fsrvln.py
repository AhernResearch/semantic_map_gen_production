"""Compatibility entrypoint for FSRVLN scene-package view conversion."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from adapters.scripts.fsrvln.scene_adapter.cli import main, parse_args  # noqa: E402,F401
from adapters.scripts.fsrvln.scene_adapter.conversion import convert  # noqa: E402,F401
from adapters.scripts.fsrvln.scene_adapter.policy import (  # noqa: E402,F401
    C_HM3DSEM,
    SOURCE_VIEW_SELECTION_VERSION,
    clear_view_payload,
    filter_depth_to_canonical_floor,
    invalidate_hmsg_checkpoints,
    resolve_canonical_floor,
    reusable_view_matches,
    source_data_signature,
    view_input_fingerprint,
)


if __name__ == "__main__":
    main()
