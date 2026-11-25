"""Monetary loss bases and welfare weights of component accounting.

Ordinary transfers cancel across their parties; proceeds of harmful
appropriation do not offset victims' losses, a value judgment whose
sensitivity is the ``net_resource_cost`` view, which cancels them too. These
weights measure monetary harm only, not a complete utility or AI benefit-cost
model.
"""

from typing import Literal

EconomicNature = Literal[
    "resource_cost", "injury_valuation", "transfer", "appropriated_transfer"
]

LAB_LOSS_BASIS = "incident_linked_ultimate_lab_cost_before_recoveries"


def welfare_weight(nature: str) -> float:
    """Deterministic accounting coefficient; never elicited from a model."""
    if nature == "transfer":
        return 0.0
    if nature in {"resource_cost", "injury_valuation", "appropriated_transfer"}:
        return 1.0
    raise ValueError(f"Unknown economic nature: {nature!r}")
