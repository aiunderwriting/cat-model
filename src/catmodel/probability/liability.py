"""Sample the insured developer's share of gross event loss with a zero-inflated Beta.

``mean`` is the unconditional share; uniform-based sampling is elementwise-pure
for chunked processing. Every parameter comes from an explicitly authored payment
kernel (``NativeLiabilityParameters``); this module supplies no default values.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import betaincinv, expit


@dataclass
class LiabilityAttenuation:
    """Liability attenuation factor for a scenario.

    Decomposes into P(recovery) × E[share | recovery]:

      recovery_rate: P(plaintiff recovers anything) — Bernoulli gate.
      mean: unconditional expected developer fault share (= recovery_rate ×
            conditional_mean).
      conditional_mean: E[developer_share | recovery] — Beta distribution
            parameter, computed as mean / recovery_rate.

    Sampling draws from a zero-inflated Beta:
      - With prob (1 - recovery_rate): attenuation = 0 (no recovery)
      - With prob recovery_rate: attenuation ~ Beta(alpha, beta), where
        alpha = conditional_mean * concentration and
        beta = (1 - conditional_mean) * concentration

    When ``severity_slope > 0`` and ``median_severity`` is set,
    ``sample_conditional_from_uniforms()`` varies the unconditional mean with
    severity:
      eff_unconditional = clip(
          mean + severity_slope * log10(max(severity, 1) / median_severity),
          0.01,
          0.99,
      )
      eff_conditional = clip(eff_unconditional / recovery_rate, 0.01, 0.99)

    When ``regulatory_recovery_uplift > 0``, the Bernoulli recovery gate
    becomes severity-dependent via a capped logistic ramp:
      eff_recovery = min(
          recovery_rate + uplift * expit(
              log10(max(severity, 1)) - log10(threshold)
          ),
          1.0,
      )
    Steepness is fixed at 1.0 in log10-space (12%-88% range spans 4 orders
    of magnitude centered on the threshold). The uplift encodes the assumption
    that large losses near and above the configured threshold attract regulatory
    enforcement, class actions, and litigation funding that increase P(any
    recovery) regardless of underlying tort theory. The uplift affects only the
    recovery gate, not the conditional mean.

    The sampled value represents the fraction of gross loss borne by
    the developer (insured). Values are 0 (no recovery) or drawn from
    the Beta (plaintiff recovers, developer pays this share).
    """

    mean: float  # Unconditional expected developer fault share ([0.01, 0.99])
    concentration: float  # Beta distribution concentration (higher = tighter)
    recovery_rate: float  # P(plaintiff recovers) — Bernoulli gate
    severity_slope: float  # mean shift per log10-unit of severity
    regulatory_recovery_threshold: float  # severity at the uplift's midpoint
    regulatory_recovery_uplift: float  # uplift amplitude before the 1.0 cap
    median_severity: float | None = None  # reference point for severity conditioning
    _log10_threshold: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.01 <= self.mean <= 0.99:
            raise ValueError(f"mean must be in [0.01, 0.99], got {self.mean}")
        if self.concentration <= 0:
            raise ValueError(
                f"concentration must be positive, got {self.concentration}"
            )
        if not 0.0 <= self.severity_slope <= 0.1:
            raise ValueError(
                f"severity_slope must be in [0, 0.1], got {self.severity_slope}"
            )
        if not 0.0 < self.recovery_rate <= 1.0:
            raise ValueError(
                f"recovery_rate must be in (0, 1], got {self.recovery_rate}"
            )
        if self.mean >= self.recovery_rate:
            raise ValueError(
                f"mean ({self.mean}) must be < recovery_rate "
                f"({self.recovery_rate}); "
                f"unconditional mean must be strictly less than P(recovery)"
            )
        if self.mean / self.recovery_rate > 0.99:
            raise ValueError(
                f"mean/recovery_rate ({self.mean / self.recovery_rate:.4f}) "
                f"exceeds 0.99; conditional_mean would be clamped, "
                f"so recovery_rate * conditional_mean would diverge from mean. "
                f"Use mean <= 0.99 * recovery_rate."
            )
        if not 0.0 <= self.regulatory_recovery_uplift <= 1.0:
            raise ValueError(
                f"regulatory_recovery_uplift must be in [0, 1], "
                f"got {self.regulatory_recovery_uplift}"
            )
        if self.regulatory_recovery_threshold <= 0:
            raise ValueError(
                f"regulatory_recovery_threshold must be positive, "
                f"got {self.regulatory_recovery_threshold}"
            )
        self._log10_threshold = float(np.log10(self.regulatory_recovery_threshold))

    @property
    def conditional_mean(self) -> float:
        return min(max(self.mean / self.recovery_rate, 0.01), 0.99)

    def sample_conditional_from_uniforms(
        self,
        severities: np.ndarray,
        recovery_u: np.ndarray,
        share_u: np.ndarray,
    ) -> np.ndarray:
        """Transform separate recovery/share uniforms into conditional factors.

        Uniform vectors must match ``severities``, lie in ``[0, 1)``, and come
        from independent streams. The transform is elementwise-pure, so chunks
        match full batches.
        """

        effective_recovery, alpha, beta = self._conditional_parameters(severities)
        recovery_uniforms = np.asarray(recovery_u, dtype=float)
        share_uniforms = np.asarray(share_u, dtype=float)
        expected_shape = effective_recovery.shape
        for name, uniforms in (
            ("recovery_u", recovery_uniforms),
            ("share_u", share_uniforms),
        ):
            if (
                uniforms.shape != expected_shape
                or np.any(~np.isfinite(uniforms))
                or np.any((uniforms < 0.0) | (uniforms >= 1.0))
            ):
                raise ValueError(
                    f"{name} must be a finite uniform vector in [0, 1) "
                    "matching severities"
                )

        recovered = recovery_uniforms < effective_recovery
        factors = np.zeros(effective_recovery.size, dtype=float)
        if np.any(recovered):
            factors[recovered] = betaincinv(
                alpha[recovered],
                beta[recovered],
                share_uniforms[recovered],
            )
        return factors

    def _conditional_parameters(
        self, severities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        severity = np.asarray(severities, dtype=float)
        if severity.ndim != 1 or np.any(~np.isfinite(severity)) or np.any(severity < 0):
            raise ValueError("severities must be a finite nonnegative vector")
        if self.regulatory_recovery_uplift > 0:
            log10_severity = np.log10(np.maximum(severity, 1.0))
            uplift_fraction = expit(log10_severity - self._log10_threshold)
            effective_recovery = np.minimum(
                self.recovery_rate + self.regulatory_recovery_uplift * uplift_fraction,
                1.0,
            )
        else:
            effective_recovery = np.full(severity.size, self.recovery_rate)

        if self.severity_slope > 0 and self.median_severity is not None:
            log_ratio = np.log10(np.maximum(severity, 1.0) / self.median_severity)
            effective_unconditional = np.clip(
                self.mean + self.severity_slope * log_ratio,
                0.01,
                0.99,
            )
            # The regulatory uplift changes P(recovery), not the recovered
            # share, so the authored base recovery rate remains the divisor.
            conditional_mean = np.clip(
                effective_unconditional / self.recovery_rate,
                0.01,
                0.99,
            )
        else:
            conditional_mean = np.full(severity.size, self.conditional_mean)
        alpha = conditional_mean * self.concentration
        beta = (1.0 - conditional_mean) * self.concentration
        return effective_recovery, alpha, beta
