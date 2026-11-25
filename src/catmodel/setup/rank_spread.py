"""The public-study reference tail form of a count law read at a rank spread,
POOL-SPREAD.

The count law itself, read at a stretched score with a post-stretch tail form,
is ``catmodel.elicitation.epistemic``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catmodel.contracts.incident import SpreadTailForm

REFERENCE_SPREAD_TAIL_FORM: SpreadTailForm = "calibrated_p95_peak"
"""The public-study reference tail form after applying the rank spread."""
