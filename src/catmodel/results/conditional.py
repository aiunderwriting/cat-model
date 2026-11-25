"""Integer kernel evaluation on retained native draws.

A kernel mixes each of its states' years of every view by one weight. A view
that counts the lab's own loss, a lab-loss view or a gross view, gives a state
left out of those views weight zero (``POOL-EXCLUDED-SAMPLES``;
``JointState.lab_loss_left_out``). A kernel that weighs such a state
therefore holds every other view and none of those, and
``ConditionalCapture.lab_loss_unavailable_reason`` says so.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from catmodel.contracts.conditional import ConditionalContext
from catmodel.contracts.incident import counts_lab_loss
from catmodel.contracts.run_setup import RunSetup
from catmodel.results.execution import (
    RunCapture,
    arm_weights,
    load_run_capture,
    validate_captured_laws,
)
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
    ConditionalCapabilityDraws,
)
from catmodel.simulation.incident_runtime import state_key


@dataclass(frozen=True)
class ConditionalCapture:
    setup: RunSetup
    capture_directory: Path
    draws: Mapping[str, Mapping[str, ConditionalCapabilityDraws]] = field(init=False)
    capture: RunCapture = field(init=False)

    def __post_init__(self) -> None:
        draws, capture = load_run_capture(self.capture_directory)
        object.__setattr__(
            self,
            "draws",
            MappingProxyType(
                {arm: MappingProxyType(dict(rows)) for arm, rows in draws.items()}
            ),
        )
        object.__setattr__(self, "capture", capture)
        validate_captured_laws(
            self.setup,
            self.draws,
            self.capture,
            arm_weights(self.setup, None),
            require_target_coverage=False,
            validate_band=False,
        )

    def context(self, context_id: str) -> ConditionalContext:
        for context in self.setup.conditional_contexts:
            if context.context_id == context_id:
                return context
        raise ValueError("unsupported_context: no authored conditional context")

    def _left_out(self, weights: Mapping[str, float]) -> list[str]:
        """The states with weight in a kernel that are left out of the
        views that count the lab's loss, in setup order."""
        return [
            state.state_id
            for state in self.setup.joint_states
            if state.lab_loss_left_out and weights.get(state.state_id, 0.0) > 0
        ]

    def lab_loss_unavailable_reason(
        self, context_id: str, eci: int, arm_id: str
    ) -> str | None:
        """Why the kernel at ``eci`` holds no view that counts the lab's
        loss, or None where it holds them."""
        kernel = self.context(context_id).kernel_at(eci, arm_id)
        left_out = self._left_out(
            {row.state_id: row.weight for row in kernel.components}
        )
        if not left_out:
            return None
        return (
            "The kernel weighs "
            + ", ".join(repr(state) for state in left_out)
            + ", left out of the views that count the lab's own loss because "
            "a case it reads declares that loss unavailable "
            "(POOL-EXCLUDED-SAMPLES). A kernel mixes each state's years of every "
            "view by one weight, so it holds the other views at the authored "
            "weights and neither the lab-loss views nor the gross views."
        )

    def _mixture(
        self, context: ConditionalContext, arm_id: str, weights: Mapping[str, float]
    ) -> CapabilityMixture:
        # A kernel that weighs a left-out state holds no view that counts
        # the lab's loss.
        without_lab_loss = bool(self._left_out(weights))
        peers = {"welfare"} | (
            set() if without_lab_loss else {f"lab_loss::{context.lab_id}"}
        )
        views = {
            view
            for view in self.setup.execution.requested_views
            if not (without_lab_loss and counts_lab_loss(view))
        }
        selected = {}
        arm = self.draws.get(arm_id, {})
        for state_id, weight in weights.items():
            if weight <= 0:
                continue
            key = state_key(state_id)
            draw = arm.get(key)
            if draw is None:
                raise ValueError(
                    "conditional distribution unavailable: missing capture"
                )
            if draw.joint_law_semantics != "physical_joint":
                raise ValueError(
                    "conditional distribution requires physical joint samples"
                )
            if not peers <= set(draw.views):
                raise ValueError(
                    "conditional distribution unavailable: societal or lab loss is missing"
                )
            views.intersection_update(draw.views)
            selected[key] = draw
        return CapabilityMixture(
            {
                key: replace(
                    draw,
                    views={
                        view: value
                        for view, value in draw.views.items()
                        if view in views
                    },
                )
                for key, draw in selected.items()
            },
            {
                state_key(state): weight
                for state, weight in weights.items()
                if weight > 0
            },
        )

    def kernel_at(self, context_id: str, eci: int, arm_id: str) -> CapabilityMixture:
        """Evaluate the authored kernel; a zero marginal forecast mass is irrelevant."""
        context = self.context(context_id)
        kernel = context.kernel_at(eci, arm_id)
        return self._mixture(
            context, arm_id, {row.state_id: row.weight for row in kernel.components}
        )
