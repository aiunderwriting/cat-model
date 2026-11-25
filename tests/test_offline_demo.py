"""The offline demo takes a small study from questions to a summary on
synthetic answers, without a provider."""

import csv
import glob
import json

import pytest

from catmodel.evidence import source_inputs
from catmodel.setup.public_study import public_study_collections
from catmodel.setup.store import load_setup
from catmodel.worktree import CodeProvenance
from scripts import offline_demo
from scripts.offline_demo import main


def _no_provider(monkeypatch) -> None:
    """No provider key, and no route to any host."""
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "META_API_KEY",
        "OPENROUTER_API_KEY",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(name, "http://127.0.0.1:9")


def test_the_demo_takes_a_synthetic_study_from_questions_to_a_summary(
    tmp_path, monkeypatch, capfd
):
    # The demo builds its source bundle from a clean committed checkout; a
    # test run's checkout need not be one.
    monkeypatch.setattr(
        source_inputs,
        "resolve_code_provenance",
        lambda _: CodeProvenance("a" * 40, True, "clean", None),
    )
    _no_provider(monkeypatch)
    output = tmp_path / "demo"

    assert main([str(output)]) == 0

    printed = capfd.readouterr().out
    # Said before the first step and again beside the results.
    assert printed.count(offline_demo.SYNTHETIC) == 2
    assert "say nothing about risk" in offline_demo.SYNTHETIC
    listed = [
        line.split(":", 1)[1].strip()
        for line in printed.split(offline_demo.SYNTHETIC)[-1].splitlines()
        if line.startswith("  ")
    ]
    assert len(listed) == 8
    for path in listed:
        assert glob.glob(path), path

    # One scenario at three interior nodes and the no-progress node, for each
    # source status, asked of one sample of one fake respondent.
    plan = output / "public-study" / "public-study.json"
    captures = [capture for _, capture in public_study_collections(plan)]
    assert len(captures) == 8
    for capture in captures:
        answers = json.loads((capture / "contextual-risk.json").read_bytes())
        (case,) = answers["cases"]
        assert case["node_id"] == offline_demo.DEFAULT_SCENARIO
        assert case["respondent"]["model_id"] == offline_demo.MODEL_ID
        report = json.loads(
            (output / "validation" / f"{capture.name}.json").read_bytes()
        )
        assert report["complete_cases"] == 1
        assert report["unanswered_cases"] == report["exhausted_calls"] == []

    # The setup's name says what its answers are, and so does the summary.
    setup = load_setup(output / "setup")
    assert setup.manifest.setup_id == offline_demo.SETUP_ID
    assert "synthetic" in offline_demo.SETUP_ID
    summary = (output / "summary" / "summary.md").read_text(encoding="utf-8")
    assert f"`{offline_demo.SETUP_ID}`" in summary
    with (output / "summary" / "summary.csv").open(
        encoding="utf-8", newline=""
    ) as rows:
        (eal,) = (
            row
            for row in csv.DictReader(rows)
            if (row["section"], row["view"], row["statistic"])
            == ("exceedance", "welfare", "expected_annual_loss")
        )
    assert eal["status"] == "available"
    assert float(eal["value"]) > 0


def test_the_help_says_the_answers_are_synthetic(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "Every answer is synthetic" in text
    assert "say nothing about risk" in text


def test_the_demo_writes_only_to_a_new_directory_outside_the_checkout(tmp_path, capsys):
    assert main([str(tmp_path)]) == 2
    assert "already exists" in capsys.readouterr().err

    inside = offline_demo.REPO_ROOT / "offline-demo-output"
    assert main([str(inside)]) == 2
    assert "inside the source checkout" in capsys.readouterr().err
    assert not inside.exists()
