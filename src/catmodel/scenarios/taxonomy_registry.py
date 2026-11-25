"""Registry of reporting domains.

Population placement belongs to the loss taxonomy; domains are derived
reporting views, not routing authority. The hierarchy contract may assign only
these domains, and results list them in this order.
"""

from typing import Literal, get_args

RiskDomainName = Literal[
    "cyber",
    "eo",
    "cbrn",
    "agent",
    "ai_rd",
    "societal",
    "svc",
    "regulatory",
]

RISK_DOMAIN_ORDER: tuple[str, ...] = get_args(RiskDomainName)

RISK_DOMAIN_SET: frozenset[str] = frozenset(RISK_DOMAIN_ORDER)


DOMAIN_SORT_RANK: dict[str, int] = {
    domain: rank for rank, domain in enumerate(RISK_DOMAIN_ORDER)
}
