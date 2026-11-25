"""Unexpected execution faults cannot publish a partial study run."""

import pytest

from catmodel.results import execution
from test_support.study import _package, _plan


@pytest.mark.parametrize("failing_arm", ["reference", "contained"])
def test_unexpected_model_fault_aborts_without_an_output(
    tmp_path, monkeypatch, failing_arm
):
    package = _package(tmp_path, plan=_plan(32))
    original = execution.simulate_incident_state

    def broken(plan, compiled, **kwargs):
        if compiled.alternative_id == failing_arm:
            raise ValueError("unexpected numerical defect")
        return original(plan, compiled, **kwargs)

    monkeypatch.setattr(execution, "simulate_incident_state", broken)
    output = tmp_path / "uncreated"
    with pytest.raises(ValueError, match="unexpected numerical defect"):
        execution.run_setup(
            setup_path=package.root, output=output, code_commit="fixture"
        )
    assert not output.exists()
