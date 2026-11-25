"""Assumption-register challenge catalog; a frozen plan declares actual coverage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from catmodel.contracts.incident import ComparisonRequirement
from catmodel.scenarios.sensitivity_coverage import (
    TAIL_SHAPE_SENSITIVITY_STEP,
)

SensitivityKind = Literal["assumption", "numerical", "diagnostic", "exhibit"]
RangeBasis = Literal[
    "elicited",
    "authored",
    "structural_alternative",
    "judgment",
    "numerical",
]


@dataclass(frozen=True, slots=True)
class SensitivityFamily:
    """One catalogued challenge to assumption register rows.

    ``kind`` separates assumption perturbations from numerical refinements,
    labelled diagnostics and exhibits. ``range_basis`` states where the size of
    the perturbation comes from: respondents' elicited ranges, values authored
    in scenario YAML, a structural alternative without a magnitude, a
    judgmental magnitude owned by code, whose status the register's Required
    sensitivity field states, or numerical refinement.
    """

    family_id: str
    kind: SensitivityKind
    assumption_ids: tuple[str, ...]
    range_basis: RangeBasis
    description: str


DEFAULT_CONTEXT_BOUND_FAMILIES = (
    "public_forecast",
    "model_pooling_leave_one_out",
    "tail_shape",
    "uncapped_severity",
    "exposure_scale",
    "safeguard_level",
)
"""Families a public-study import selects when none are named.

A judgment of materiality per unit of simulation; assumptions.md gives each
family's purpose. The default stays within about five times the reference
work. Its costs:

- ``public_forecast`` and ``model_pooling_leave_one_out`` reweight the
  reference draws and add no simulation; leave-one-out is inapplicable with
  fewer than three models.
- ``tail_shape`` adds two runs and ``uncapped_severity`` one, none when no node
  requests the uncapped alternative.
- ``exposure_scale`` and ``safeguard_level`` are unavailable and free without
  variant studies. With them each variant arm simulates its own states, the variant's
  embedded-rule nodes and its no-progress node (at weight zero) times its
  selections: two three-node variants at three samples per model have
  2 x 4 x 3 = 24 node-samples per model against the headline's 8 x 5 = 40
  (seven interior nodes and the no-progress node at five samples), about 60%
  more states for each family.

The analytic ``large_loss_anchor`` exhibit challenges ``SEV-LARGE-LOSS-ANCHOR``
in every contextual result at no simulation cost: its EAL alternatives ignore the
anchor, scale every shaping share's conditional odds above p99 by one third
and three, and pass every law through its stated very-large-loss share (the
two-anchor tail). The simulated ``large_loss_anchor`` family (two runs), the
``very_large_loss_anchor`` family (one run) and the
``large_loss_anchor_ignored`` diagnostic (one run) would take the default set
past five times. ``lab_loss_cap`` (three runs) and ``lab_loss_provider_only``
(one) move the lab-loss view and, through the annual business loss, gross,
but not welfare, which the tornado ranks: select them whenever lab loss is
reported. ``severity_cap`` adds two runs (the
uncapped alternative already bounds its higher endpoint from above),
``count_family`` and ``severity_endpoint_stress`` one each, and
``model_pooling`` none. ``epistemic_spread_calibration`` adds the fixed
k = 1.5 and 0.75 alternatives, ``epistemic_tail_width`` two runs and
``common_frequency_bias`` two. Every family not selected is frozen as an
unavailable requirement.
"""

LARGE_LOSS_ANCHOR_FACTORS = (1 / 3, 3.0)
"""Factors on the conditional odds, given an incident above the law's own
p99, that it reaches the large-loss level (``SEV-LARGE-LOSS-ANCHOR``).

The ``INITIAL_ESTIMATE`` alternative size of about one order of magnitude across the
alternatives is a chosen stress with no calibration record. Scaling the odds
keeps both alternatives feasible for every shaping share and symmetric in log-odds.
"""

COUNT_FAMILY_VARIANCE_TO_MEAN = 2.0
"""Variance-to-mean ratio of the negative-binomial count alternatives (``FREQ-POISSON``).

An INITIAL_ESTIMATE judgment of mild overdispersion at fixed mean: a rare
origin then averages 1/ln 2, about 1.44, incidents in a year with any. No
matched incident series measures it.
"""

EPISTEMIC_TAIL_FACTORS = (0.75, 1.5)
"""Narrower/wider extrapolated epistemic tails; five quantiles fixed.

An INITIAL_ESTIMATE stress range with no coverage interpretation: no
respondent calibration record identifies these factors. They multiply the
capped local tail widths beyond p5/p95, so 1.5 deliberately exceeds the base
cap and shows how much a mean rests on it, separately from whole-law rank
spreading.
"""

EXPOSURE_SCALE_VARIANTS = {"half": 0.5, "double": 2.0}
"""Factors of the ``exposure_scale`` variants, an INITIAL_ESTIMATE judgment
(``POP-HYPOTHETICAL-LAB``).

Each variant multiplies every path-linked deployment scale at every trajectory
node, before rounding to two significant figures, so it moves every scale
together.
"""

SAFEGUARD_LEVELS = ("weaker", "stronger")
"""Levels of the ``safeguard_level`` variants (``POP-SAFEGUARD-LEVEL``).

Each level replaces the study's stipulated safeguard texts
(``SAFEGUARD_LEVEL_FIELDS``) with one coherent posture authored in the study
config: ``weaker`` a plausible less careful frontier developer, ``stronger``
the strictest safeguards that the roster labs' published safety frameworks
describe, applied before the capability thresholds they are attached to.
"""

FRONTIER_LAB_VALUE_USD = 1e12
"""INITIAL_ESTIMATE enterprise value of a frontier AI lab during the study period.

Each episode's lab loss from provider-borne components plus compensation, and
separately each node's annual business loss, is capped at this value, or at
the scenario's tail-class endpoint where that is lower. One stipulated value
serves every lab, named or hypothetical; ``SEV-LAB-CAP`` gives the
limited-liability rationale and the weak grounding. The cap moves the
lab-loss view and, through the annual business loss, gross; welfare is
unchanged.
"""
FRONTIER_LAB_VALUE_SENSITIVITY_USD = (3e11, 3e12)
"""Lower and higher frontier-lab values of the ``lab_loss_cap`` alternatives.

An INITIAL_ESTIMATE judgment of about a factor of three either side of the
base: the lower alternative stands for a smaller frontier lab or a value that falls in
the catastrophe itself, the higher alternative for valuation growth to the policy
period or a lab inside one of the largest listed groups. A third alternative removes
the lab cap, so only the tail-class endpoint remains where one exists.
"""


def usd_text(value: float) -> str:
    """A round USD amount in words, such as ``USD 300 billion``."""
    if value >= 1e12:
        return f"USD {value / 1e12:g} trillion"
    return f"USD {value / 1e9:g} billion"


EPISTEMIC_SPREAD_FACTORS = (1.5, 0.75)
"""Uncalibrated widened and narrowed spread factors ``k``.

No setup reads a respondent fit, so these fixed alternatives are the spread
challenge (``POOL-SPREAD``).
Every epistemic rank ``u`` is read at ``Phi(k Phi^-1(u))``, so a stated 90%
interval (p5 to p95) covers ``2 Phi(1.645 / k) - 1``: about 73% at ``k = 1.5``
and 97% at ``k = 0.75``. Medians and rank order are unchanged. An
INITIAL_ESTIMATE judgment: human interval studies document undercoverage that
varies by format and domain, which motivates a widening stress but identifies
no factor for these respondents. Ranks beyond p5 and p95 use the judged local
tails, capped after the stretch (``REFERENCE_SPREAD_TAIL_FORM``).
"""

COMMON_FREQUENCY_BIAS_FACTORS = (0.5, 2.0)
"""Factors for a pool-wide multiplicative frequency bias (``POOL-COMMON-BIAS``).

Chosen INITIAL_ESTIMATE factors, not set from the respondent fit's
estimate of b (the register row gives that estimate's limits). Each factor
multiplies all five count quantiles and their expected annual count; severity
laws, branch probabilities, shares and exposure stay fixed.
"""

SENSITIVITY_FAMILIES: tuple[SensitivityFamily, ...] = (
    SensitivityFamily(
        "epistemic_tail_width",
        "assumption",
        ("POOL-PCHIP",),
        "judgment",
        "Capped extrapolated epistemic tail widths multiplied by {:g} and {:g} "
        "at k = 1 and b = 1, preserving all five count quantiles and the "
        "central PCHIP law.".format(*EPISTEMIC_TAIL_FACTORS),
    ),
    SensitivityFamily(
        "tail_shape",
        "assumption",
        ("SEV-QUANTILE-LAW", "SEV-GPD-SHAPE"),
        "judgment",
        f"GPD shape {TAIL_SHAPE_SENSITIVITY_STEP:g} lower and higher on every "
        "severity law with a GPD component. A law keeps its knots and p99 and "
        "re-derives the GPD scale, which also sets the body's p95 slope, so its "
        "quantiles between p75 and p95 move too. A local one-step sensitivity, "
        "not an uncertainty range.",
    ),
    SensitivityFamily(
        "large_loss_anchor",
        "assumption",
        ("SEV-LARGE-LOSS-ANCHOR",),
        "judgment",
        "Every large-loss share that shapes a law has its conditional odds above "
        "p99 multiplied by {:.4g} and {:g}, keeping which cases the anchor "
        "shapes. Simulated, opt-in; the analytic exhibit reports the same alternatives' "
        "EAL effects in every contextual result.".format(*LARGE_LOSS_ANCHOR_FACTORS),
    ),
    SensitivityFamily(
        "very_large_loss_anchor",
        "assumption",
        ("SEV-LARGE-LOSS-ANCHOR",),
        "elicited",
        "Every law whose stated very-large-loss share is positive and coherent "
        "also passes its tail through that share, the two-anchor tail: a second "
        "power law from the large-loss level to the very-large-loss level and "
        "the governed tail conditioned on reaching it, or the only anchor above "
        "the law's own p99 where the large-loss level lies within the stated "
        "quantiles. No reference law reads that share. Simulated, opt-in; the "
        "analytic exhibit reports the same alternative's EAL effect in every "
        "contextual result.",
    ),
    SensitivityFamily(
        "large_loss_anchor_ignored",
        "diagnostic",
        ("SEV-LARGE-LOSS-ANCHOR",),
        "structural_alternative",
        "Every law compiled with the p99-only tail, the large-loss anchor "
        "ignored: a design comparison, not an assumption challenge or a lower "
        "bound, so it is reported but not ranked. Simulated, opt-in.",
    ),
    SensitivityFamily(
        "uncapped_severity",
        "assumption",
        ("SEV-CAP",),
        "structural_alternative",
        "Finite cap removed at the governed shape, for scenarios whose "
        "severity governance requests the uncapped stress alternative.",
    ),
    SensitivityFamily(
        "severity_cap",
        "assumption",
        ("SEV-CAP",),
        "authored",
        "Authored finite-cap endpoints, or 75% and 125% of the cap where "
        "severity governance authors none.",
    ),
    SensitivityFamily(
        "severity_endpoint_stress",
        "assumption",
        ("SEV-CAP",),
        "authored",
        "Authored finite endpoint applied to an uncapped severity posture.",
    ),
    SensitivityFamily(
        "lab_loss_cap",
        "assumption",
        ("SEV-LAB-CAP", "ACC-LAB-ANNUAL-BUSINESS-LOSS"),
        "judgment",
        "The stipulated frontier-lab value that caps one episode's lab loss, "
        "provider-borne components plus compensation, and separately one node's "
        "annual business loss, at {} and {} instead of "
        "{}, and removed so each keeps only the tail-class endpoint where "
        "one exists and is otherwise uncapped.".format(
            *map(usd_text, FRONTIER_LAB_VALUE_SENSITIVITY_USD),
            usd_text(FRONTIER_LAB_VALUE_USD),
        ),
    ),
    SensitivityFamily(
        "lab_loss_provider_only",
        "assumption",
        ("ACC-LAB-BEARER", "ACC-LAB-ANNUAL-BUSINESS-LOSS"),
        "structural_alternative",
        "Each episode's lab loss kept to its provider-borne components, with the "
        "elicited lab compensation share and annual business loss set to zero: "
        "the lower bound of the lab-loss rule, under the same episode cap.",
    ),
    SensitivityFamily(
        "model_pooling_leave_one_out",
        "assumption",
        ("POOL-WEIGHTS",),
        "structural_alternative",
        "The whole analysis with each model left out in turn and the others "
        "equally weighted, when at least three models answered.",
    ),
    SensitivityFamily(
        "epistemic_spread_calibration",
        "assumption",
        ("POOL-SPREAD",),
        "judgment",
        "Read every count-law rank at the uncalibrated k = {:g} and {:g}, with "
        "the reference tail form, the medians and the five stated count "
        "quantiles fixed. The respondent-calibration fit of k and a common "
        "level bias b is a diagnostic that no setup reads; these chosen "
        "factors are not set from it.".format(*EPISTEMIC_SPREAD_FACTORS),
    ),
    SensitivityFamily(
        "common_frequency_bias",
        "assumption",
        ("POOL-COMMON-BIAS",),
        "judgment",
        "Every node's five count quantiles and expected annual count multiplied "
        "by the chosen INITIAL_ESTIMATE factors {:g} and {:g}; severity "
        "laws, branch probabilities, shares and exposure stay fixed. EAL moves "
        "linearly by the same factors; tail statistics are recomputed. The "
        "respondent-calibration fit reports b_hat, its ranges and log k–log b "
        "correlation, plus marginal b at k_ref, as a diagnostic; the "
        "chosen factors are not set from them. Transfer from known-count "
        "seeds is unvalidated, and most seed windows share one quarter.".format(
            *COMMON_FREQUENCY_BIAS_FACTORS
        ),
    ),
    SensitivityFamily(
        "count_family",
        "assumption",
        ("FREQ-TERMINAL", "FREQ-POISSON"),
        "judgment",
        "Negative-binomial repeated counts and episode arrivals with "
        f"variance-to-mean {COUNT_FAMILY_VARIANCE_TO_MEAN:g} at fixed mean.",
    ),
    SensitivityFamily(
        "model_pooling",
        "assumption",
        ("POOL-WEIGHTS", "SEV-ONE-LAW"),
        "structural_alternative",
        "The whole analysis conditioned on each model in turn.",
    ),
    SensitivityFamily(
        "exposure_scale",
        "assumption",
        ("POP-HYPOTHETICAL-LAB", "FREQ-CHAIN-EXPOSURE"),
        "judgment",
        "Every path-linked deployment scale multiplied by {} at each node of the "
        "embedded trajectory rule, each variant a separately collected public study, "
        "measured against the headline captures reweighted to the same embedded "
        "rule.".format(
            " and ".join(f"{factor:g}" for factor in EXPOSURE_SCALE_VARIANTS.values())
        ),
    ),
    SensitivityFamily(
        "safeguard_level",
        "assumption",
        ("POP-SAFEGUARD-LEVEL",),
        "authored",
        "The lab's stipulated safeguards (scope, permissions and oversight) "
        "replaced by the study config's {} levels at each node of the embedded "
        "trajectory rule, each variant a separately collected public study, measured "
        "against the headline captures reweighted to the same embedded "
        "rule.".format(" and ".join(SAFEGUARD_LEVELS)),
    ),
    SensitivityFamily(
        "public_forecast",
        "assumption",
        ("OBS-PUBLIC-JOINT", "OBS-PUBLIC-TIMING"),
        "structural_alternative",
        "Alternative public-frontier forecasts integrated on a context-bound "
        "setup's trajectory quadrature nodes by their exact node weights: the "
        "pace-shape alternatives, the lower pace cap, the trend-discrepancy "
        "interval ends and no progress, each with independent lab paces.",
    ),
)
# Requirements needing additional evidence or an explicitly authored alternative.
UNAVAILABLE_ASSUMPTION_SENSITIVITIES: Mapping[str, str] = {
    "OBS-PUBLIC-JOINT": "A pace cap above the forecast's, or none, has paths beyond every node, so it needs a study planned and collected under that cap. Correlated or shared lab paces need new paired captures, and window and prior alternatives need authored inputs.",
    "OBS-PUBLIC-TIMING": "A finer trajectory rule needs its additional nodes collected, so the embedded lower-order rule is the integration check. More representative paths per node need new paired captures.",
    "DEP-PUBLIC-CONTEXT": "Public-path independence comparisons are offline; alternative residual couplings require separately authored conditional loss laws.",
    "POP-UNIVERSE": "A provider add/drop comparison requires a separately captured compatible universe and study.",
    "POP-HYPOTHETICAL-LAB": "Fixed scales, or a growth ceiling of 2 or 5, need their own collected public study; none is generated.",
    "POP-SAFEGUARD-LEVEL": "Levels that vary scope, permissions or oversight one at a time, or that pair a safeguard level with an exposure variant, need their own collected public study. A level that changes who reviews an output or action needs questions that ask the scenarios it moves incidents between together.",
    "POP-HISTORY-UNITS": "Population, unit and threshold alternatives require complete new annual responses.",
    "POP-ADMISSION": "Each admission alternative defines a different population and requires complete new frequency and severity answers.",
    "POP-WHOLE-INCIDENT": "Linked parts across leaves with shared-origin dependence are a different taxonomy and require complete new answers.",
    "POP-PRIVATE-CLAIMS": "A course-of-conduct matter unit or a different boundary with commercial contract disputes defines a different population and requires complete new annual responses.",
    "POP-TERMINAL-INCIDENT": "An alternative incident definition defines a different population and requires complete new annual responses.",
    "ACC-ATTRIBUTION": "Provider allocation and ownership alternatives require separately captured compatible inputs.",
    "ACC-ORIGIN": "Common-occurrence and nested-ownership alternatives are not implemented.",
    "ACC-COMPOSITION": "Contextual nodes elicit composition as partition shares; alternative or size-dependent composition requires new captures.",
    "ACC-LAB-BEARER": "No arm generates a business loss comonotone with the societal severity rank or a full-compensation bound. Only public-study setups that select it generate the provider-only bound. Another bearer classification needs new captures.",
    "ACC-LAB-ANNUAL-BUSINESS-LOSS": "A business loss that rises with the year's incident count, or a per-incident business loss, needs a new question and new captures. Only public-study setups that select them generate the provider-only bound and the lab-cap alternatives.",
    "ACC-SCOPE": "No loading for unmodeled populations or excluded harms is implemented; omitted scope is disclosed, not quantified.",
    "OBS-ECI-SUFFICIENCY": "A second capability coordinate requires an observable series, a forecast and new paired captures.",
    "OBS-RESPONSE-FORM": "No route generates response-form alternatives; an authored setup declares its own as alternative arms.",
    "OBS-CONTEXT-CONTINUATION": "A wider anchor hull or a continuation beyond it changes the frozen question and requires a separately collected instrument.",
    "OBS-CAUSAL": "A reference-held response needs new paired captures on a held trajectory; respondent repeatability, framing and held-out diagnostics have no collection protocol.",
    "POOL-PCHIP": "Fixed-anchor contextual setups execute only their reference arm, so only public-study setups generate the tail-width arms; global lognormal parameter-law fits are not implemented.",
    "FREQ-MONETARY-BASIS": "Alternative monetary bases or growth adjustments require compatible new answers.",
    "SEV-QUANTILE-LAW": "Severity-location and p99-excess alternatives are not generated; splice-threshold and winsorization alternatives are not implemented.",
    "SEV-CAP": "A cap or endpoint challenge requires an authored outer cap or finite endpoint and a transform consistent with the retained qualification population; missing lower-threshold count/mark inputs are not inferred.",
    "SEV-LAB-CAP": "Fixed-anchor contextual setups execute only their reference arm, so only public-study setups that select the family generate the lab-cap alternatives.",
    "SEV-LARGE-LOSS-ANCHOR": "Only public-study setups that select the families simulate the anchor alternatives and the two-anchor tail; fixed-anchor setups execute only their reference arm. Every contextual result carries the analytic large_loss_anchor exhibit.",
    "POOL-SHARED-MODEL": "Respondents are matched across nodes and source statuses; independent-respondent coupling and comonotone node ranks are not generated.",
    "POOL-COMMON-BIAS": "Fixed-anchor contextual setups execute only their reference arm, so they do not generate common frequency-bias alternatives.",
    "POOL-EXCLUDED-SAMPLES": "No result with the excluded samples exists, because the answers they lack do not, and no lab-loss or gross view with a left-out sample, because its lab-loss answer does not. Two computable alternatives are not generated: a reweighting that excludes each excluded respondent at every node, and one that leaves each left-out sample out of every view.",
    "POOL-UNANSWERED-SCENARIOS": "No result uses a model's own answers to a scenario it does not answer: the answers its samples lack do not exist, and the rest are a selected subset. Another donor assignment is computable from the same captures but not generated.",
    "FREQ-EVENT-CHAIN": "An alternative chain structure is a different chain question and requires new chain answers.",
    "FREQ-CHAIN-EXPOSURE": "An alternative exposure base, factor or derivation changes the chain question and requires new chain answers.",
    "HIER-REMAINDER": "The remainder has no count or severity law; simulating it needs authored populations for the unestimated leaves.",
    "HIER-THRESHOLD": "Another admission threshold is a different parent question and leaf population that need new answers.",
    "DEP-RESIDUAL-FACTOR": "Gamma-intensity stresses are not generated; an authored setup declares them with matched controls.",
    "SEV-EMPIRICAL-TRANSFER": "Source-to-target transfer alternatives require captured source and population fits.",
    "DEP-CONDITIONAL-INDEPENDENCE": "Independent-respondent coupling, comonotone node ranks and count-dependence stresses are not generated; additional dependence or common-occurrence alternatives require captured complete joint laws.",
    "FIN-POLICY-TERMS": "Occurrence-mapping, retention and limit alternatives require separately captured compatible packages.",
    "FIN-TRIGGER": "Claims-made reporting windows, retroactive dates and extended reporting periods are not implemented; another temporal basis would need a new setup field and a separately authored compatible population.",
    "FIN-NOMINAL-TERMS": "Claim-inflation and settlement-delay alternatives are not implemented; deflated terms require a separately authored compatible setup.",
    "FIN-LIABILITY": "No study alternative varies liability parameters; each alternative requires a separately authored compatible setup, and collected evidence has no finance input.",
    "NUM-BUDGETS": "Other process-year budgets and work allowances are not executed as study alternatives; each is a separate import through the import functions' `years_per_state`, `max_reference_state_years`, `max_comparison_state_years` and `work_allowances`.",
}
CONTEXT_BOUND_UNAVAILABLE_CHALLENGES: tuple[str, ...] = (
    "OBS-CAUSAL",
    "OBS-ECI-SUFFICIENCY",
    "ACC-COMPOSITION",
    "ACC-LAB-BEARER",
    "ACC-LAB-ANNUAL-BUSINESS-LOSS",
    "POOL-PCHIP",
    "POOL-SHARED-MODEL",
    "POOL-EXCLUDED-SAMPLES",
    "SEV-QUANTILE-LAW",
    "DEP-CONDITIONAL-INDEPENDENCE",
    "DEP-RESIDUAL-FACTOR",
    "NUM-BUDGETS",
)
"""Register rows whose required challenge no context-bound setup executes.

Every context-bound plan freezes each of them, with the route-specific rows
below, as an unavailable comparison requirement keyed by its register ID and
carrying its reason from ``UNAVAILABLE_ASSUMPTION_SENSITIVITIES``. Results
therefore list these challenges beside the catalog families, and comparison
coverage cannot read complete while any of them is unavailable.
"""
PUBLIC_STUDY_UNAVAILABLE_CHALLENGES: tuple[str, ...] = (
    *CONTEXT_BOUND_UNAVAILABLE_CHALLENGES,
    "OBS-PUBLIC-JOINT",
    "OBS-PUBLIC-TIMING",
    "DEP-PUBLIC-CONTEXT",
)
FIXED_ANCHOR_UNAVAILABLE_CHALLENGES: tuple[str, ...] = (
    *CONTEXT_BOUND_UNAVAILABLE_CHALLENGES,
    "POOL-COMMON-BIAS",
    "OBS-CONTEXT-CONTINUATION",
    "SEV-LAB-CAP",
    "SEV-LARGE-LOSS-ANCHOR",
)


def unavailable_challenge_requirements(
    register_ids: Sequence[str], scenario_ids: Sequence[str]
) -> tuple[ComparisonRequirement, ...]:
    """Freeze unavailable register challenges over a setup's scenario scope."""
    return tuple(
        ComparisonRequirement(
            family_id=register_id,
            scenario_ids=tuple(scenario_ids),
            status="unavailable",
            reason=UNAVAILABLE_ASSUMPTION_SENSITIVITIES[register_id],
            held_fixed=("The captured questions, answers and compiled laws.",),
        )
        for register_id in register_ids
    )


# Register checks that do not require a separate sensitivity alternative.
REGISTER_CHECKS_WITHOUT_STUDY_ALTERNATIVE: Mapping[str, str] = {
    "ACC-WELFARE": "Where component composition identifies both views, report net_resource_cost beside welfare; otherwise disclose the unavailable view. The two bases are accounting conventions, not uncertainty bounds.",
    "FREQ-PARAMETER-PROCESS": "A dimensional rule with no alternative value, which the analytic oracles check.",
    "SEV-INFINITE-VARIANCE": "A mathematical property of the GPD; results suppress unsupported dollar variance and precision instead of varying it.",
    "NUM-PRECISION": "Seeded tests of estimator bias and coverage and of quadrature certificates assess the numerical claim. None establishes model or forecast accuracy.",
    "NUM-STREAMS": "Chunk-boundary invariance and pairing with the base are software properties that regression tests check.",
    "NUM-TAIL-GATE": "Every gated statistic reports its relative half-width, so availability at another tolerance can be read from the output; other caps need a rerun with the import options. Seeded tests in `tests/test_tail_gate.py` check the half-widths, withholding, doubling, replay and interval coverage.",
    "NUM-SMALL-EVENTS": "The audit control `small_events.body = individual` draws every small event, and regression tests in `tests/test_small_events.py` compare its annual totals and tails with the summed run. The reported envelope bounds the TVaR difference without simulation.",
    "NUM-SEVERITY-DOMAIN": "A numerical admission rule; laws and components outside the domain are refused at their contracts, which regression tests check.",
    "DEP-MIXTURE": "A total-probability identity with no alternative value; alternative forecasts reweight unchanged stored states, and draw-budget changes never alter weights.",
    "FREQ-CHAIN-INDEPENDENCE": "No alternative coupling is reported: it would need a dependence parameter that no question elicits. The frequency_estimate exhibit reports the reference EAL under the chain-only count law.",
    "FREQ-CHAIN-POOL": "The frequency_estimate exhibit reports the reference EAL under the direct-only (weight 0) and chain-only (weight 1) counts, which bracket the pool weight 0.5. Tail statistics would need simulated arms, which are not generated.",
    "FREQ-CHAIN-GUARDRAIL": "The frequency_estimate exhibit reports how the retries ended, and its before_chain_guardrail arm gives the reference EAL with every chain answer as first given. Looser thresholds can be computed from stored calls; stricter ones need new retry calls.",
    "HIER-CATEGORY-POOL": "The frequency_estimate exhibit reports the reference EAL at the two ends of the category answer's weight: before_hierarchy (weight 0) and hierarchy_parents_only (weight 1 at every parent). Across levels the run's EAL need not lie between the two arms. Tail statistics would need simulated arms, which are not generated.",
    "FREQ-EAL-CROSSCHECK": "The frequency_estimate exhibit's before_eal_crosscheck arm reports the reference EAL with the count and law before the cross-check. A higher factor can be computed from stored calls by reverting the cases it would not call; a lower one needs new cross-check calls.",
    "NUM-CALL-LIMITS": "An execution control. `capability-risk validate` reports each model's output-limit stops and generation times; another budget, ceiling or deadline needs a new run directory and fresh answers.",
    "NUM-CONNECTION-LOST": "An execution control. `capability-risk validate` reports each model's `connection_lost` re-sends; another re-send bound needs a new run directory and fresh answers.",
    "NUM-PROVIDER-FAILURES": "An execution control. A re-sent provider failure selects nothing the respondent said, and every failed attempt stays in the capture. Another bound needs only a changed constant.",
    "NUM-PLAUSIBILITY": "A report-only reading aid. Every share and count is reported, so any other threshold can be applied to the same tables.",
    "NUM-RESPONDENT-SAMPLING": "A report-only exhibit. Regression tests in `tests/test_respondent_sampling.py` check it against a hand calculation and a stratified delete-one jackknife.",
    "NUM-CAPABILITY-SCENARIOS": "A deterministic report-only exhibit, tested in `tests/test_capability_scenarios.py` against direct sampling of independent lab paths. Other percentiles recompute from the retained base law without new collection or simulation.",
    "NUM-EAL-PARAMETER-BAND": "A report-only exhibit. Regression tests in `tests/test_eal_parameter_band.py` check that a fixed rate collapses it to a point and that replay accepts a recomputation only within its bounds and tolerance.",
}
