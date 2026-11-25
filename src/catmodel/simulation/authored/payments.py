"""Payment kernels: the reduced-form share of a component a developer pays.

Only a hand-authored setup carries a payment kernel (``NativePaymentKernel``);
a setup imported from collected answers has none. A direct component is paid
whole, and a liability-subject component is paid the kernel's liability share
(``catmodel.probability.liability``), drawn at the severity rank of the
kernel's source law when the share depends on severity
(``incident_linear.draw_single_member_events``,
``incident_campaigns.draw_episodes``).
"""

from __future__ import annotations

from catmodel.contracts.components import (
    NativePaymentKernel,
    native_payment_uses_severity,
)


def native_payment_positive_mean_lower_bound(
    kernel: NativePaymentKernel, component_id: str
) -> float:
    """Lower bound on the native proxy's mean paid fraction, uniform in severity.

    Zero when the kernel never pays the component.
    """
    if component_id in kernel.direct_component_ids:
        return 1.0
    if component_id not in kernel.liability_component_ids:
        return 0.0
    if kernel.liability is None:
        return 1.0
    if kernel.liability.mean == 0 or kernel.liability.recovery_rate == 0:
        return 0.0
    if kernel.liability.mean == 1:
        return 1.0
    return kernel.liability.recovery_rate * 0.01


def constant_native_payment_fraction(kernel: NativePaymentKernel) -> float | None:
    """Return the liability fraction when it does not depend on severity."""
    if (
        kernel.liability_component_ids
        and kernel.liability is not None
        and (kernel.liability.mean == 0 or kernel.liability.recovery_rate == 0)
    ):
        return 0.0
    return None if native_payment_uses_severity(kernel) else 1.0
