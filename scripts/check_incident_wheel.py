"""Install a wheel with frozen dependencies and, with network access blocked, run
stipulated examples with it outside the checkout.

Outside a checkout the commands record the installed version as the code's
identity, which each example's setup and the campaign example's run are
checked to carry.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_GUARD = """import socket
def blocked(*args, **kwargs):
    raise AssertionError("network forbidden during installed-wheel acceptance")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
socket.getaddrinfo = blocked
"""

_ACCEPTANCE = """
import hashlib, json, math, os
import numpy as np
from importlib.metadata import version
from pathlib import Path
import catmodel
from catmodel.contracts.incident import ExecutionPlan
from catmodel.setup.store import load_setup
from catmodel.resources import config_path, source_checkout_root
from catmodel.scenarios.coverage_register import load_coverage_register
from catmodel.scenarios.hierarchy_contract import load_hierarchy_contract
from catmodel.scenarios.loader import load_scenarios
from catmodel.scenarios.taxonomy_identity import load_scenario_lineage_register
from catmodel.resources import scenarios_path
from catmodel.results.execution import load_run_capture
from catmodel.results.results_export import export_results, load_results_export
from catmodel.results.results_file import load_results
from catmodel.results.conditional import ConditionalCapture

assert source_checkout_root() is None, catmodel.__file__
assert not any("API_KEY" in key for key in os.environ)
assert load_hierarchy_contract().nodes
assert load_coverage_register().gaps
catalog = json.loads(scenarios_path("scenario_catalog.json").read_bytes())
assert catalog["scenarios"]
assert (
    load_scenario_lineage_register().current_taxonomy_version
    == catalog["meta"]["scenario_taxonomy_version"]
)
assert {row.id for row in load_scenarios(scenarios_path())} == {
    entry["id"] for entry in catalog["scenarios"]
}
root = Path.cwd()
source = config_path("setups", "stipulated-campaign")
plan = ExecutionPlan.model_validate_json((source / "plan.json").read_bytes())
code = os.environ["CAT_ACCEPTANCE_COMMIT"]
def invoke(*args):
    import subprocess, sys
    subprocess.run([str(Path(sys.executable).with_name("catmodel")), *map(str, args)],
                   check=True, capture_output=True, text=True)

setup = root / "setup"
invoke("setup", "build", "--spec", source / "setup.json",
       "--plan", source / "plan.json", "--setup-id", "installed-wheel-example",
       "--evidence", source / "evidence", "--output", setup)
package = load_setup(setup)
# The installed wheel has no checkout: a setup and a run record its version
# and no commit.
built = package.manifest.code_provenance
assert built["tree_state"] == "installed", built
assert built["commit"] is None and built["version"] == version("catmodel")
invoke("run", "--setup", setup, "--output", root / "run")
report = load_results(root / "run" / "results.json", diagnostics=True)
installed = report["producer_provenance"]
assert installed["tree_state"] == "installed", installed
assert installed["commit"] is None and installed["version"] == version("catmodel")
capture = root / "run" / "capture"
before = {str(p.relative_to(capture)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in capture.rglob("*") if p.is_file()}
invoke("results", "replay", "--setup", setup, "--capture", capture,
       "--output", root / "replay")
replay = load_results(root / "replay" / "results.json", diagnostics=True)
assert replay["diagnostics"] == report["diagnostics"]
assert report["arms"] == replay["arms"]
export_results(setup_path=setup, capture_directory=capture,
               results_path=root / "run/results.json", output=root / "export")
assert load_results_export(root / "export").format == "cat.results/1"
conditional_source = config_path("setups", "stipulated-conditional")
conditional_setup = root / "conditional-setup"
invoke("setup", "build", "--spec", conditional_source / "setup.json",
       "--plan", conditional_source / "plan.json", "--setup-id", "conditional-wheel-example",
       "--output", conditional_setup)
conditional_package = load_setup(conditional_setup)
assert conditional_package.manifest.code_provenance["tree_state"] == "installed"
invoke("run", "--setup", conditional_setup, "--output", root / "conditional-run")
conditional_capture = root / "conditional-run/capture"
conditional_export = export_results(
    setup_path=conditional_setup, capture_directory=conditional_capture,
    output=root / "conditional-export",
)
assert load_results_export(root / "conditional-export") == conditional_export
assert conditional_export.conditioning["status"] == "available"
kernel = ConditionalCapture(conditional_package, conditional_capture).kernel_at("internal", 1, "reference")
assert abs(kernel.metrics("welfare")["expected_annual_loss"] - 70) < 1e-10
assert abs(kernel.metrics("lab_loss::lab")["expected_annual_loss"] - 28) < 1e-10
weights = root / "weights.json"
weights.write_text(json.dumps({"slow_fixed": .25, "fast_rising": .75}))
invoke("results", "replay", "--setup", setup, "--capture", capture,
       "--weights", weights, "--level", ".95", "--output", root / "reweighted")
weighted = load_results(root / "reweighted" / "results.json")
after = {str(p.relative_to(capture)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in capture.rglob("*") if p.is_file()}
assert before == after
draws, metadata = load_run_capture(capture)
assert set(draws) == set(plan.required_arms)
assert all(len(arm) == 2 for arm in draws.values())
assert metadata.physical_joint_semantics == "physical_joint_samples"
means = {}
for arm, expected in (("reference",171.6),("contained",54.8),("displacement",96.4)):
    metrics = report["arms"][arm]["metrics"]
    value = metrics["gross"]["expected_annual_loss"]
    assert metrics["gross"]["eal_estimator"] == "native_analytic_mean"
    assert abs(value - expected) <= max(1e-10, abs(expected) * 1e-12), (arm,value,expected)
    empirical = metrics["gross"]["normalized_empirical_mean"]
    precision = report["diagnostics"]["arms"][arm]["views"]["gross"]["mean_diagnostics"]
    assert precision["empirical_mean_numerical_interval_95"] is None
    state_weights = {row.state_id: row.weight for row in package.joint_states}
    # Batch means: 16 contiguous batches of years, the same years in each state.
    batch_means = 0.0
    for key, draw in draws[arm].items():
        assert draw.views["gross"].variance_exists
        values = draw.annual_losses["gross"]
        weight = state_weights[metadata.states[key].state_id]
        batch = np.arange(len(values)) * 16 // len(values)
        batch_means = batch_means + weight * np.bincount(batch, weights=values) / np.bincount(batch)
    empirical_se = float(np.std(batch_means, ddof=1)) / 4
    assert math.isclose(precision["empirical_mean_standard_error"], empirical_se, rel_tol=1e-9)
    assert abs(empirical - expected) <= 6 * empirical_se, (arm,empirical,expected,empirical_se)
    assert metrics["net_resource_cost"]["status"] == "unavailable"
    assert metrics["scenario::campaign_harm::gross"]["status"] == "unavailable"
    assert metrics["financial"]["status"] == "unavailable"
    means[arm] = {"reported_native_analytic_eal":value, "normalized_empirical_mean":empirical, "empirical_standard_error":empirical_se, "independent_analytic_eal":expected}
assert weighted["analysis_sha256"] != replay["analysis_sha256"]
summary = {"installed_module":catmodel.__file__, "code_commit":code,
           "code_provenance":{"setup":built, "run":installed},
           "network_blocked":True, "provider_keys_present":False,
           "all_arms_retained":True, "capture_unchanged_by_reduction":True,
           "cli_paths":["setup build", "setup inspect", "run", "results replay", "results replay --weights", "results export"],
           "conditional_query_and_export":True,
           "means":means, "capture_content_sha256":metadata.content_sha256}
(root / "acceptance.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\\n")
print(json.dumps(summary,sort_keys=True))
"""


_FINANCIAL_ACCEPTANCE = """
import hashlib,json
from pathlib import Path
import numpy as np
from catmodel.contracts.components import (ComponentAmountLaw,ComponentObligation,
    ComponentPayment,ContractTerms,PolicyMembership)
from catmodel.simulation.components import component_view_weight
from catmodel.simulation.policy_terms import PolicyReduction,policy_shares,share_parts

# This is a separately authored software oracle, not collected financial evidence.
fixture={
    "meaning":"Openly stipulated 80 indemnity plus 20 outside-limit defense; one occurrence limit 50.",
    "mark":{
        "components":[
            {"component_id":"harm","amount":80,"loss_type":"indemnity"},
            {"component_id":"defense","amount":20,"loss_type":"defense"}],
        "obligations":[
            {"component_id":"harm","entity_id":"stipulated-operator","fraction":1},
            {"component_id":"defense","entity_id":"stipulated-operator","fraction":1}],
        "payments":[
            {"component_id":"harm","entity_id":"stipulated-operator","fraction":1},
            {"component_id":"defense","entity_id":"stipulated-operator","fraction":1}],
        "memberships":[
            {"component_id":"harm","contract_id":"stipulated-policy",
             "obligated_entity_id":"stipulated-operator","eligible":True},
            {"component_id":"defense","contract_id":"stipulated-policy",
             "obligated_entity_id":"stipulated-operator","eligible":True,"outside_limits":True}]},
    "contract":{"contract_id":"stipulated-policy","contract_year":"stipulated-year",
                "entity_id":"stipulated-operator","occurrence_limit":50}}
data=(json.dumps(fixture,sort_keys=True)+"\\n").encode()
root=Path.cwd()
(root/"stipulated-financial-oracle.json").write_bytes(data)
mark=fixture["mark"]
components=tuple(ComponentAmountLaw.model_validate(row) for row in mark["components"])
contract=ContractTerms.model_validate(fixture["contract"])
shares=policy_shares(components,
    tuple(ComponentObligation.model_validate(row) for row in mark["obligations"]),
    tuple(ComponentPayment.model_validate(row) for row in mark["payments"]),
    tuple(PolicyMembership.model_validate(row) for row in mark["memberships"]),
    (),{contract.contract_id:contract})
# One year of one event: the array reduction the simulation applies.
reduction=PolicyReduction((contract,),1,financial=True,occurrence_maximum=False)
event=np.zeros(1,dtype=np.int64)
amounts=[np.array([row.amount],dtype=float) for row in components]
reduction.add(event,share_parts(reduction,shares,amounts,(),event),origin="stipulated-origin")
gross=sum(row.amount*component_view_weight(row,"gross") for row in components)
payment=float(reduction.views()["financial"][0])
assert gross==100
assert payment==70
summary=json.loads((root/"acceptance.json").read_bytes())
summary["stipulated_financial_oracle"]={
    "basis":"separately_authored_software_fixture",
    "captured_liability_evidence":False,
    "fixture_sha256":hashlib.sha256(data).hexdigest(),
    "gross":gross,"payment":payment,
    "independent_exact_payment":70}
(root/"acceptance.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\\n")
print(json.dumps(summary["stipulated_financial_oracle"],sort_keys=True))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--wheel",
        type=Path,
        help="Reuse a wheel built from this source snapshot; its checksum is recorded.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheel = None if args.wheel is None else args.wheel.expanduser().resolve()
    if wheel is not None and (not wheel.is_file() or wheel.suffix != ".whl"):
        parser.error("--wheel must name an existing wheel file")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)

    def run(command, *, cwd=root, env=None):
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            capture_output=True,
            timeout=300,
        )
        return result.stdout

    if wheel is None:
        run(["uv", "build", "--wheel", "--offline", "--out-dir", str(output / "dist")])
        wheels = tuple((output / "dist").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("wheel acceptance requires exactly one built wheel")
        wheel = wheels[0]
    requirements = output / "requirements.txt"
    run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--format",
            "requirements-txt",
            "--output-file",
            str(requirements),
        ]
    )
    environment = output / "venv"
    run(["uv", "venv", "--python", "3.13", str(environment)])
    python = environment / "bin" / "python"
    run(
        [
            "uv",
            "pip",
            "install",
            "--no-config",
            "--python",
            str(python),
            "--require-hashes",
            "-r",
            str(requirements),
        ]
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--no-config",
            "--offline",
            "--no-deps",
            "--python",
            str(python),
            str(wheel),
        ]
    )
    guard = output / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(_GUARD)
    work = output / "outside-checkout"
    work.mkdir()
    commit = run(["git", "rev-parse", "HEAD"]).strip()
    dirty = bool(run(["git", "status", "--porcelain"]).strip())
    env = {
        key: os.environ[key]
        for key in (
            "PATH",
            "LANG",
            "LC_ALL",
            "TZ",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
        if key in os.environ
    }
    env.update(
        PYTHONPATH=str(guard),
        CAT_ACCEPTANCE_COMMIT=commit + ("-dirty" if dirty else ""),
    )
    print(run([str(python), "-c", _ACCEPTANCE], cwd=work, env=env), end="")
    print(run([str(python), "-c", _FINANCIAL_ACCEPTANCE], cwd=work, env=env), end="")
    inspect = run(
        [
            str(environment / "bin" / "catmodel"),
            "setup",
            "inspect",
            "--setup",
            str(work / "setup"),
        ],
        cwd=work,
        env=env,
    )
    (work / "cli-inspection.json").write_text(inspect)
    run(
        [
            str(environment / "bin" / "catmodel"),
            "results",
            "export",
            "--setup",
            str(work / "setup"),
            "--capture",
            str(work / "run/capture"),
            "--output",
            str(work / "cli-export"),
        ],
        cwd=work,
        env=env,
    )
    summary = json.loads((work / "acceptance.json").read_bytes())
    summary.update(
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        lock_sha256=hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        wheel_build="built_from_checkout" if args.wheel is None else "provided",
    )
    (output / "acceptance.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr, file=sys.stderr)
        raise SystemExit(exc.returncode) from None
