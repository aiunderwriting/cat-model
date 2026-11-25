"""Count frozen annual simulation work without acquiring or simulating inputs."""

from collections.abc import Mapping
from typing import Any, cast

from catmodel.contracts.incident import (
    DEFAULT_WORK_ALLOWANCES,
    WORK_ALLOWANCES,
    ExecutionPlan,
    check_numerical_controls,
)
from catmodel.hashing import canonical_json_sha256
from catmodel.simulation.tail_gate import doubling_limits

# How a refusal names the count each allowance bounds.
_WORK_NAMES = {
    "vectorized_initiators": "individually drawn events",
    "campaign_members": "campaign members",
}


def work_allowances(controls: Mapping[str, object]) -> dict[str, int]:
    """A plan's work allowances per simulated year (``WORK_ALLOWANCES``), each
    at its default where the plan omits it."""
    check_numerical_controls(controls)
    return {
        name: cast(int, controls.get(name, default))
        for name, default in DEFAULT_WORK_ALLOWANCES.items()
    }


def work_limits(allowances: Mapping[str, int], years: int) -> dict[str, int]:
    """Each bounded count's limit for a state simulated for ``years`` years:
    its allowance per year times the years, keyed by the count it bounds."""
    return {
        count: allowances[allowance] * years
        for allowance, count in WORK_ALLOWANCES.items()
    }


class WorkLimitExceeded(ValueError):
    """A state's work over all its simulated years exceeds its limit.

    The message names the state (and its arm, once known), the count, the
    years, the limit, the allowance it comes from and how to raise it.
    """

    def __init__(
        self,
        allowance: str,
        per_year: int,
        state_id: str,
        count: int,
        years: int,
        arm: str | None = None,
    ) -> None:
        self.details = (allowance, per_year, state_id, count, years)
        self.arm = arm
        where = f"state {state_id!r}" + ("" if arm is None else f" of arm {arm!r}")
        super().__init__(
            f"{_WORK_NAMES[WORK_ALLOWANCES[allowance]]} in {where}: {count:,} over "
            f"{years:,} simulated years, above the limit of {per_year * years:,}, "
            f"{allowance} ({per_year:,} a year) times the years; they average "
            f"{count / years:,.1f} a year. If the law is intended, raise {allowance} "
            "in the plan's numerical controls and build the setup again, or import "
            "again with a larger work_allowances (import_public_frontier_study, "
            "import_contextual_setup); otherwise correct the law"
        )

    def in_arm(self, arm: str) -> "WorkLimitExceeded":
        return WorkLimitExceeded(*self.details, arm=arm)

    def __reduce__(self):
        # Rebuilt from its details, so a worker process's refusal reaches the
        # parent unchanged.
        return WorkLimitExceeded, (*self.details, self.arm)


def simulation_identity(
    plan: ExecutionPlan,
    arm: str,
    state: str,
    conditional_key: str,
    simulations: int | None = None,
) -> str:
    """Identify one conditional simulation by everything that determines it.

    Streams are seeded by the plan seed and the state, and the conditional key
    identifies the compiled conditional law, so an arm that leaves a state's law
    unchanged (for example a forecast reweighting) has the same identity as the
    reference and reuses its draws.
    """
    resolved = plan.for_arm(arm)
    return canonical_json_sha256(
        {
            "state": state,
            "conditional_key": conditional_key,
            "simulations": resolved.simulations if simulations is None else simulations,
            "numerical_controls": resolved.numerical_controls,
        }
    )


def reference_followers(
    plan: ExecutionPlan, conditional_keys: Mapping[str, Mapping[str, str]]
) -> frozenset[str]:
    """Arms that follow the reference arm: every state they simulate is one the
    reference arm simulates with the same simulation identity, as when an arm
    only reweights the reference arm's states. They reuse its draws, so they
    take its years and cost no simulation.
    """
    reference = plan.reference_arm
    own = {
        state: simulation_identity(plan, reference, state, key)
        for state, key in conditional_keys[reference].items()
    }
    return frozenset(
        arm
        for arm in plan.required_arms
        if arm != reference
        and all(
            own.get(state) == simulation_identity(plan, arm, state, key)
            for state, key in conditional_keys[arm].items()
        )
    )


def execution_budget(
    plan: ExecutionPlan, conditional_keys: Mapping[str, Mapping[str, str]]
) -> dict:
    """Count requested and actually simulated conditional years by arm.

    ``conditional_keys`` maps each arm to its required states' conditional law
    keys. A state-year whose simulation identity an earlier arm (the reference
    first) already has is reused, not simulated again. Each arm also reports
    the most years per state, and state-years, that the tail gate may double
    it to within its cap; an arm that follows the reference arm takes the
    reference arm's (``reference_followers``). Each arm reports its states'
    work limits at its most years.
    """
    simulated: set[str] = set()
    followers = reference_followers(plan, conditional_keys)
    limits = doubling_limits(
        plan,
        {arm: len(conditional_keys[arm]) for arm in plan.required_arms},
        followers,
    )
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in (
        plan.reference_arm,
        *(arm for arm in plan.required_arms if arm != plan.reference_arm),
    ):
        resolved = plan.for_arm(arm)
        keys = conditional_keys[arm]
        source = plan.reference_arm if arm in followers else arm
        largest = plan.for_arm(source).simulations * 2 ** limits[arm]
        new = {
            simulation_identity(plan, arm, state, key) for state, key in keys.items()
        } - simulated
        simulated |= new
        by_arm[arm] = {
            "conditional_states": len(keys),
            "years_per_state": resolved.simulations,
            "conditional_years": len(keys) * resolved.simulations,
            "simulated_conditional_years": len(new) * resolved.simulations,
            "follows_arm": plan.reference_arm if arm in followers else None,
            "max_years_per_state": largest,
            "max_conditional_years": len(keys) * largest,
            "max_work_per_state": work_limits(
                work_allowances(resolved.numerical_controls), largest
            ),
        }
    reference = by_arm[plan.reference_arm]["simulated_conditional_years"]
    total = sum(row["simulated_conditional_years"] for row in by_arm.values())
    return {
        "arms": len(by_arm),
        "by_arm": by_arm,
        "simulated_conditional_years": total,
        "reused_conditional_years": sum(
            row["conditional_years"] for row in by_arm.values()
        )
        - total,
        "reference_conditional_years": reference,
        "reference_work_multiple": total / reference,
        "cost_basis": "Frozen state-year count of each arm's required states; arms that leave a state's conditional law unchanged reuse its draws. The tail gate may double an arm's years up to its maximum, where each state's work may reach its allowances per year times those years, and an arm that follows the reference arm takes its years. Realized event counts, runtime and dollar cost are not inferred from it.",
    }
