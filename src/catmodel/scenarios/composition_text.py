"""Shared accounting prose for broad severity-composition populations.

A composition cell may use a descriptive ``{{accounting:name}}`` as its whole
``loss_form`` or ``rationale``; loader validation expands it before rendering
or serialization. The shared rosters in ``scenarios/composition_rosters.yaml``
use it, and a scenario's own cells may too. Named and narrower populations keep
their specific text inline; this table does not choose cells or weights.
"""

ACCOUNTING_TEXT: dict[str, str] = {
    "external_resources": (
        "External-party repair, care, restoration, investigation, response and lost "
        "productive output after avoided costs and replacement supply; exclude "
        "deployer/provider resources and separately valued injury. Exclude displaced "
        "profit or market share gained by other standing parties; that is an ordinary "
        "transfer, not resource destruction."
    ),
    "deployer_resources": (
        "Non-provider deployer/manufacturer correction, recall, repair, defense, "
        "response and lost productive output after avoided costs and replacement "
        "supply, excluding other bearers. Exclude displaced profit or market share "
        "gained by other standing parties; that is an ordinary transfer, not resource "
        "destruction."
    ),
    "provider_resources": (
        "Selected-provider investigation, defense, remediation, productive-asset "
        "impairment and lost productive output after avoided costs and replacement "
        "supply, without duplicating capitalized cash flows or costs borne by others. "
        "Exclude displaced profit or market share gained by other standing parties; "
        "that is an ordinary transfer, not resource destruction."
    ),
    "direct_injury": (
        "Directly affected people's compensatory injury or fatality valuation, "
        "excluding separately counted care, lost output and compensation payments; "
        "not VSL."
    ),
    "external_ordinary_transfers": (
        "External affected party's ordinary trading or investment transfers, "
        "erroneous or denied payments retained by non-appropriating recipients, and "
        "distinct contractual or noncivil penalties within this occurrence scope. "
        "Exclude theft, fraud or coercive appropriation, separately listed civil "
        "fines, resource costs, injury valuation and compensation for already-counted "
        "harm. Recipient gains cancel in aggregate welfare; the payer's financial "
        "loss remains. Include displaced profit or IP rents gained by other standing "
        "parties within this population; exclude harmfully appropriated value."
    ),
    "deployer_ordinary_transfers": (
        "Non-provider deployer or manufacturer's ordinary trading or investment "
        "transfers, erroneous or denied payments retained by non-appropriating "
        "recipients, and distinct contractual or noncivil penalties within this "
        "occurrence scope. Exclude theft, fraud or coercive appropriation, separately "
        "listed civil fines, resource costs, injury valuation and compensation for "
        "already-counted harm. Recipient gains cancel in aggregate welfare; the "
        "payer's financial loss remains. Include displaced profit or IP rents gained "
        "by other standing parties within this population; exclude harmfully "
        "appropriated value."
    ),
    "provider_ordinary_transfers": (
        "Selected provider's ordinary trading or investment transfers, erroneous or "
        "denied payments retained by non-appropriating recipients, and distinct "
        "contractual or noncivil penalties within this occurrence scope. Exclude "
        "theft, fraud or coercive appropriation, separately listed civil fines, "
        "resource costs, injury valuation and compensation for already-counted harm. "
        "Recipient gains cancel in aggregate welfare; the payer's financial loss "
        "remains. Include displaced profit or IP rents gained by other standing "
        "parties within this population; exclude harmfully appropriated value."
    ),
    "external_appropriation": (
        "External affected party's funds or assets appropriated through theft, fraud "
        "or coercion, including ransom, within this occurrence scope, before recovery "
        "of principal. Classify the harmful act, not the recipient's organizational "
        "identity; do not credit perpetrators' proceeds. Exclude ordinary trading "
        "counterparty gains, government fines, compensation for already-counted harm "
        "and separately counted response costs. This cell permits a zero share."
    ),
    "deployer_appropriation": (
        "Non-provider deployer or manufacturer's funds or assets appropriated through "
        "theft, fraud or coercion, including ransom, within this occurrence scope, "
        "before recovery of principal. Classify the harmful act, not the recipient's "
        "organizational identity; do not credit perpetrators' proceeds. Exclude "
        "ordinary trading counterparty gains, government fines, compensation for "
        "already-counted harm and separately counted response costs. This cell "
        "permits a zero share."
    ),
    "provider_appropriation": (
        "Selected provider's funds or assets appropriated through theft, fraud or "
        "coercion, including ransom, within this occurrence scope, before recovery of "
        "principal. Classify the harmful act, not the recipient's organizational "
        "identity; do not credit perpetrators' proceeds. Exclude ordinary trading "
        "counterparty gains, government fines, compensation for already-counted harm "
        "and separately counted response costs. This cell permits a zero share."
    ),
    "ordinary_transfers_rationale": (
        "Ordinary transfers with recipient standing cancel in aggregate welfare, "
        "while payer losses remain in financial views. Harmful appropriation has its "
        "own cell; an undifferentiated transfer total does not identify the "
        "appropriation share. The roster's shares are elicited, not authored. "
        "Include displaced profit or IP rents gained "
        "by other standing parties within this population; exclude harmfully "
        "appropriated value."
    ),
    "appropriation_rationale": (
        "Victim deprivation counts without treating perpetrators' proceeds as a "
        "social benefit. Its magnitude is an uncollected composition share; no "
        "empirical share is asserted."
    ),
    "provider_civil_fine_rationale": (
        "Provider-borne civil-fine transfer attributed to this origin. Its "
        "magnitude is this component's composition share: uncollected in an "
        "estimated base composition, or stated by an explicitly selected structural "
        "case. Coverage classifies and filters this component separately."
    ),
}


def expand_accounting_text(value: str) -> str:
    """Expand one explicit whole-field reference; leave literal prose intact."""
    prefix = "{{accounting:"
    if prefix not in value:
        return value
    if not value.startswith(prefix) or not value.endswith("}}"):
        raise ValueError("accounting references must occupy the whole text field")
    name = value[len(prefix) : -2]
    try:
        return ACCOUNTING_TEXT[name]
    except KeyError as exc:
        raise ValueError(f"unknown accounting text reference: {name!r}") from exc
