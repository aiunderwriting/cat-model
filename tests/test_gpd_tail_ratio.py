import numpy as np
import pytest

from catmodel.probability.distributions import (
    Distribution,
    gpd_scale_from_tail_ratio,
)

BODY = dict(p5=1.0, p25=5.0, p50=20.0, p75=60.0, p95=200.0)


def _ratio(d: Distribution) -> float:
    return d.quantile(0.99) / d.quantile(0.95)


class TestResolver:
    @pytest.mark.parametrize("xi", [0.3, 0.5, 0.65, -0.3, 1e-12])
    @pytest.mark.parametrize("R", [1.5, 2.85, 5.0])
    def test_closed_form_hits_ratio(self, xi, R):
        sigma = gpd_scale_from_tail_ratio(u_thr=200.0, xi=xi, tail_ratio=R, p_thr=0.95)
        d = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=xi, gpd_scale=sigma
        )
        assert _ratio(d) == pytest.approx(R, rel=1e-6)

    @pytest.mark.parametrize("R", [1.0, 0.9, 0.0])
    def test_ratio_must_exceed_one(self, R):
        with pytest.raises(ValueError, match="must be > 1"):
            gpd_scale_from_tail_ratio(u_thr=200.0, xi=0.5, tail_ratio=R, p_thr=0.95)

    def test_requires_positive_splice(self):
        with pytest.raises(ValueError, match="positive splice"):
            gpd_scale_from_tail_ratio(u_thr=0.0, xi=0.5, tail_ratio=2.0, p_thr=0.95)


class TestDistributionAnchor:
    def test_anchor_reaches_ratio_continuity_cannot(self):
        cont = Distribution(severity_minimum_strict=False, **BODY, gpd_shape=0.5)
        anchored = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=0.5, gpd_tail_ratio=2.85
        )
        assert _ratio(cont) < 2.85
        assert _ratio(anchored) == pytest.approx(2.85, rel=1e-6)
        assert anchored.quantile(0.999) > cont.quantile(0.999)

    def test_deep_tail_still_moves_with_xi(self):
        low_xi = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=0.4, gpd_tail_ratio=2.85
        )
        high_xi = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=0.65, gpd_tail_ratio=2.85
        )
        assert _ratio(low_xi) == pytest.approx(2.85, rel=1e-6)
        assert _ratio(high_xi) == pytest.approx(2.85, rel=1e-6)
        assert high_xi.quantile(0.999) > low_xi.quantile(0.999)


class TestValidation:
    def test_ratio_without_shape_rejected(self):
        with pytest.raises(ValueError, match="require gpd_shape"):
            Distribution(severity_minimum_strict=False, **BODY, gpd_tail_ratio=2.0)

    def test_scale_and_ratio_mutually_exclusive(self):
        with pytest.raises(ValueError, match="at most one"):
            Distribution(
                severity_minimum_strict=False,
                **BODY,
                gpd_shape=0.5,
                gpd_scale=100.0,
                gpd_tail_ratio=2.0,
            )

    @pytest.mark.parametrize("bad", [1.0, 0.5])
    def test_ratio_must_exceed_one(self, bad):
        with pytest.raises(ValueError, match="> 1"):
            Distribution(
                severity_minimum_strict=False, **BODY, gpd_shape=0.5, gpd_tail_ratio=bad
            )


class TestPropagation:
    def test_shape_change_preserves_ratio_and_re_solves_sigma(self):
        d = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=0.5, gpd_tail_ratio=2.85
        )
        sigma0 = d.effective_gpd_scale
        p99 = float(d.sample_from_uniforms(np.array([0.99]))[0])
        d2 = d.with_gpd_shape(0.6)
        assert d2.gpd_tail_ratio == pytest.approx(2.85)
        assert d2.effective_gpd_scale != pytest.approx(sigma0)
        assert d2.gpd_shape == pytest.approx(0.6)
        assert _ratio(d2) == pytest.approx(2.85, rel=1e-6)
        assert float(d2.sample_from_uniforms(np.array([0.99]))[0]) == pytest.approx(p99)
        assert float(d2.sample_from_uniforms(np.array([0.999]))[0]) != pytest.approx(
            float(d.sample_from_uniforms(np.array([0.999]))[0])
        )

    def test_excess_scale_preserves_scale_free_ratio(self):
        d = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=0.5, gpd_tail_ratio=2.85
        )
        d3 = d.with_excess_scale(3.0)
        assert _ratio(d3) == pytest.approx(2.85, rel=1e-6)
