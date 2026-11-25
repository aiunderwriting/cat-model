# Readable-summary snapshot

`golden-poisson-summary.md` is the `summary.md` that the [README quickstart](../../../README.md#quickstart) writes for the [golden Poisson example](../../../config/setups/golden-poisson/README.md): the summary of its replay at the default tail levels and 0.95 and 0.98, with the 12-digit analysis digest in its header replaced by `<analysis>`. The digest covers the reducer's provenance, whose Python, platform and package versions differ between CI runners. `tests/test_golden_poisson.py` runs the example through the CLI and compares its summary with this file, so a change to the summary's layout, wording, statuses or rounding, or to the seeded run, shows up as a reviewed diff.

Regenerate it only for a reviewed, intentional change, and say in the pull request what changed and why. From the repository root, with new output directories:

```sh
uv run --frozen catmodel setup build \
  --spec config/setups/golden-poisson/setup.json \
  --plan config/setups/golden-poisson/plan.json \
  --setup-id golden-poisson --output /tmp/golden-setup
uv run --frozen catmodel run --setup /tmp/golden-setup --output /tmp/golden-run
uv run --frozen catmodel results replay --setup /tmp/golden-setup \
  --capture /tmp/golden-run/capture --level 0.95 --level 0.98 \
  --output /tmp/golden-replay
uv run --frozen catmodel results summary \
  --results /tmp/golden-replay/results.json --output /tmp/golden-summary
sed -E 's/(\*\*Analysis:\*\* `)[0-9a-f]{12}(`)/\1<analysis>\2/' \
  /tmp/golden-summary/summary.md > tests/fixtures/summary/golden-poisson-summary.md
```
