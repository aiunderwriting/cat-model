"""A small valid respondent-spread fit for contract tests."""

from __future__ import annotations

from catmodel.contracts.respondent_calibration import (
    BootstrapSummary,
    CalibrationCaptureIdentity,
    CalibrationTreatments,
    CommonBiasRange,
    CookeDiagnostics,
    CookeScore,
    CurationSensitivityDiagnostics,
    JointGrid,
    JointRange,
    LeaveOneBlockOutDiagnostics,
    MarginalBiasFit,
    MarginalFits,
    MarginalSpreadFit,
    PITAtSpread,
    PoissonSpreadAlternative,
    PooledPhiSpreadAlternative,
    PrimaryPITDiagnostics,
    RespondentSeedFitRow,
    RespondentSpreadDiagnostics,
    RespondentSpreadFit,
    RespondentSpreadObservation,
    SeedPIT,
    SpreadObservationAlternatives,
    SpreadRange,
    SuperBlockBootstrapDiagnostics,
)


def respondent_spread_fit(treatments: CalibrationTreatments) -> RespondentSpreadFit:
    """Build a schema-valid fit with caller-supplied captured treatments."""
    k_hat, low, high = 1.5, 1.2, 1.8
    b_hat, b_low, b_high = 1.0, 0.8, 1.2
    digest = "a" * 64
    pit = PITAtSpread(
        median=0.5,
        central_50_coverage=1.0,
        central_90_coverage=1.0,
        histogram_10=(0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        seeds=(SeedPIT(seed_id="seed", block_id="block", pit=0.5),),
    )
    cooke = CookeScore(
        observed_proportions=(0, 0, 0, 0, 1, 0),
        expected_proportions=(0.05, 0.2, 0.25, 0.25, 0.2, 0.05),
        divergence=0,
        statistic=0,
        degrees_of_freedom=5,
        score=1,
    )
    joint_range = JointRange(
        k=SpreadRange(low=low, high=high),
        b=CommonBiasRange(low=b_low, high=b_high),
        correlation=0.0,
    )
    diagnostics = RespondentSpreadDiagnostics(
        primary_pit=PrimaryPITDiagnostics(
            at_k_1_b_1=pit,
            at_k_hat_b_1=pit,
            at_k_1_b_hat=pit,
            at_k_hat_b_hat=pit,
        ),
        leave_one_block_out=LeaveOneBlockOutDiagnostics(
            joint_total_gain=1,
            conditional_k_total_gain=1,
            blocks=(),
        ),
        cooke=CookeDiagnostics(
            at_k_1_b_1=cooke,
            at_k_hat_b_1=cooke,
            at_k_1_b_hat=cooke,
            at_k_hat_b_hat=cooke,
        ),
        super_block_bootstrap=SuperBlockBootstrapDiagnostics(
            range=joint_range, super_blocks=2
        ),
        without_curation_sensitive_seeds=CurationSensitivityDiagnostics(
            seed_ids=(), unavailable_reason="No curation-sensitive seeds are listed."
        ),
        per_model=(),
        controls_and_secondary_pit=(),
        memorization=(),
    )
    return RespondentSpreadFit(
        schema_version=1,
        status="accepted",
        capture=CalibrationCaptureIdentity(
            instrument_sha256=digest,
            collection_plan_sha256=digest,
            artifact_sha256=digest,
            resolutions_sha256=digest,
        ),
        treatments=treatments,
        tail_form="calibrated_p95_peak",
        k_bounds=(0.5, 4.0),
        b_bounds=(0.1, 10.0),
        grid=JointGrid(
            k_points=64,
            b_points=101,
            k_coarse_stride=3,
            b_coarse_stride=5,
        ),
        observation=RespondentSpreadObservation(
            model_name="negative_binomial_2",
            pooled_phi=0.0,
            minimum_periods=6,
            fallback_seed_ids=(),
        ),
        k_hat=k_hat,
        range=SpreadRange(low=low, high=high),
        b_hat=b_hat,
        b_range=CommonBiasRange(low=b_low, high=b_high),
        k_b_correlation=0.0,
        marginal=MarginalFits(
            k_at_b_1=MarginalSpreadFit(
                k_hat=k_hat,
                range=SpreadRange(low=low, high=high),
            ),
            b_at_reference_k=MarginalBiasFit(
                reference_spread=k_hat,
                reference_spread_basis="accepted_k_hat",
                b_hat=b_hat,
                range=CommonBiasRange(low=b_low, high=b_high),
            ),
        ),
        bootstrap=BootstrapSummary(replicates=2000, seed=42, blocks=2),
        alternatives=SpreadObservationAlternatives(
            poisson=PoissonSpreadAlternative(
                k_hat=k_hat,
                range=SpreadRange(low=low, high=high),
                b_hat=b_hat,
                b_range=CommonBiasRange(low=b_low, high=b_high),
            ),
            pooled_phi=PooledPhiSpreadAlternative(
                phi=0.0,
                k_hat=k_hat,
                range=SpreadRange(low=low, high=high),
                b_hat=b_hat,
                b_range=CommonBiasRange(low=b_low, high=b_high),
            ),
        ),
        diagnostics=diagnostics,
        seeds=(
            RespondentSeedFitRow(
                seed_id="seed",
                block_id="block",
                role="primary",
                value=1,
                phi=0.0,
                phi_periods=6,
                phi_source="history",
                phi_fallback_reason=None,
                pit_at_k_1_b_1=0.5,
                pit_at_k_hat_b_hat=0.5,
            ),
        ),
    )
