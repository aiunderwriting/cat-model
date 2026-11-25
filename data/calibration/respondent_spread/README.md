# Respondent spread calibration seeds

This package holds known-answer count questions for the respondent spread calibration, which compares the respondents' count judgments with resolved outcomes and estimates a spread factor `k` and a common level bias `b` ([methodology](../../../methodology.md#respondent-spread-calibration); commands in the [runbook](../../../runbook.md#respondent-spread-calibration)). The fit is a diagnostic, and no setup reads it.

## Files and readers

| Path | Reader | Contents |
|---|---|---|
| `seeds.yaml` | `scripts/respondent_calibration.py collect` | Seed wording, window, information date, source, evidence history, latest training cutoff and frozen resolution protocol. It contains no resolved outcomes. |
| `resolutions.yaml` | `scripts/respondent_calibration.py fit` | Values, retrieval times, snapshot hashes, resolver paths and notes. |
| `sources.yaml` | Documentation | Source URLs, license or terms, automated-access notes and snapshot identity. |
| `scripts/calibration_seeds/catalogue.yaml` | `scripts/calibration_seeds/assemble.py` | Authored seed definitions, source metadata, training cutoff and resolution rules. |
| `scripts/calibration_seeds/<source>.py` | Operator | One resolver per source; each writes a JSON result for the assembler. |
| `scripts/calibration_seeds/assemble.py` | Operator | Validates resolver outputs and writes the three YAML files above. |

The instrument presents each seed in its own context, because the histories of some primary questions include values for control windows.

## Roster

There are 43 primary seeds in 18 dependence blocks, 3 controls and 3 secondary seeds, and `resolutions.yaml` holds a value for every one. At most three primary seeds are in any block. Primary windows run from 2026-07-01 through 2026-10-31, after the latest respondent training cutoff of 2026-06-30, which is also their information date.

| Block | Primary seeds | Shared records or driver |
|---|---|---|
| V-KEV | `kev-additions-2026q3`, `kev-microsoft-additions-2026q3`, `kev-additions-2026-10` | One catalog; Microsoft additions are a subset. |
| V-CVE | `nvd-cves-published-2026q3`, `nvd-cves-published-2026-10`, `nvd-prompt-injection-cves-2026q3` | One database; prompt-injection CVEs are a subset. |
| V-ICS | `cisa-ics-advisories-2026q3`, `cisa-ics-medical-advisories-2026q3` | One advisory series. |
| V-GHSA | `ghsa-reviewed-2026q3`, `go-vulndb-reports-2026q3`, `rustsec-advisories-2026q3` | Mirrored advisory records. |
| S-MAL | `osv-mal-npm-2026q3`, `osv-mal-pypi-2026q3`, `osv-mal-crates-2026q3` | One malicious-package pipeline. |
| X-EDB | `exploitdb-additions-2026q3` | — |
| B-NOTICE | `hhs-ocr-breaches-2026q3`, `ca-ag-breach-notices-2026q3`, `wa-ag-ransomware-notices-2026q3` | Some incidents are notified to several regulators. |
| B-SEC | `sec-8k-item105-2026q3`, `sec-8k-item105-2026-10`, `sec-8k-cybersecurity-incident-2026q3` | One filing population; Item 1.05 filings use the phrase. |
| F-CFPB | `cfpb-fraud-scam-complaints-2026q3`, `cfpb-money-transfer-complaints-2026q3`, `cfpb-fraud-alert-complaints-2026q3` | One complaint database; fraud or scam complaints are mostly money-transfer complaints. |
| C-DEFI | `defihacklabs-incidents-2026q3` | — |
| D-SGO | `nhtsa-sgo-ads-reports-2026-07-08`, `nhtsa-sgo-adas-reports-2026-07-08` | One reporting order and release cycle. |
| D-NHTSA | `nhtsa-vehicle-recalls-2026q3`, `nhtsa-software-recalls-2026q3`, `nhtsa-odi-investigations-2026-07-10` | One agency office; software recalls are a subset, and investigations can lead to recalls. |
| M-FDA | `fda-device-recall-events-2026q3`, `fda-device-software-recall-events-2026q3`, `fda-drug-recall-events-2026q3` | One weekly Enforcement Report. |
| G-FR | `fedreg-ai-documents-2026q3`, `fedreg-cybersecurity-documents-2026q3` | One publication and search index. |
| E-ARXIV | `arxiv-cs-ai-2026q3`, `arxiv-cs-cr-2026q3` | One archive; categories can cross-list papers. |
| G-DOJ | `doj-cyber-press-releases-2026q3`, `doj-fraud-press-releases-2026q3` | One press office. |
| G-SECLIT | `sec-litigation-releases-2026q3`, `sec-litigation-releases-2026-10` | One release series. |
| G-OFAC | `ofac-sdn-additions-2026q3`, `ofac-sdn-additions-2026-10` | One sanctions list service. |

Controls are 2026-Q1 KEV, NVD and HHS OCR counts. Secondary seeds are maxima for the largest HHS breach, Washington notice and NHTSA vehicle recall. Both are reported and not fitted.

## Evidence vintage

Every outcome uses the source state retrieved on 2026-11-02; a later retrieval can differ after late postings, rejections, withdrawals, corrections or index changes. Every history row has a `period_end` before its seed window and on or before its information date: 2026-06-30 for primary and secondary seeds, and the day before its window for a control. Each history counts records as the source records them at retrieval, as the calibration question tells respondents, so it can include later revisions to earlier periods.

- **Git sources** (KEV through CISA's `cisagov/kev-data` mirror, GitHub Advisory Database, CISA CSAF and DeFiHackLabs) compute history at the last commit on or before the information date. The current snapshot, whose commit `sources.yaml` records, supplies the window value.
- **Sources that post records after their event date** (HHS OCR, California and Washington AG lists, CFPB, RustSec advisory dates and NHTSA SGO) stop histories one month before the information date, or at April 2026 for SGO. The guard is fixed, not a measured lag for each source.
- **Publication-dated sources** (NVD, OSV, Exploit Database, SEC filings and releases, OFAC, Federal Register, arXiv, openFDA, NHTSA recalls and investigations, and DOJ) build history from the retrieved snapshot, so later rejections, withdrawals or corrections can change counts.

`seeds.yaml` freezes the snapshot and replacement rules in `resolution_protocol`, with the SHA-256 of each resolver script. The fit refuses a resolver whose bytes differ, so even a comment edit needs a regenerated package, and it refuses a resolution without a value. A digest identifies the resolver as committed; it does not certify that those bytes ran on the retrieval date. No seed is replaced after collection starts.

## Regenerating

`seeds.yaml`, `resolutions.yaml` and `sources.yaml` are generated: each resolver writes one JSON result for its source, and `assemble.py` joins the results with the catalogue and overwrites the three files. Both procedures below need the `calibration-seeds` dependency group and `git`, and run from outside the data root and the results directory. The SEC EDGAR and litigation scripts stop if `SEC_EDGAR_USER_AGENT` is empty.

### Reassembling the committed vintage

This rebuilds the three files from the inputs retrieved on 2026-11-02 and retrieves nothing. It needs the data root and the results directory that the retrieval wrote, which the repository does not hold, unchanged and at the same paths:

```sh
ROOT=/path/to/cat-model
DATA=/path/to/archived-data
OUT=/path/to/archived-results
PY="$ROOT/.venv/bin/python"
export SEC_EDGAR_USER_AGENT="<organization> <operator contact>"

for source in cisa_kev nvd cisa_csaf ghsa osv exploitdb defihacklabs hhs_ocr ca_ag wa_ag \
              sec_edgar sec_litigation cfpb doj ofac federal_register nhtsa_sgo \
              nhtsa_defects openfda arxiv; do
  "$PY" -I "$ROOT/scripts/calibration_seeds/$source.py" \
    --data-dir "$DATA" --out "$OUT/$source.json" --offline
done

"$PY" -I "$ROOT/scripts/calibration_seeds/assemble.py" --results-dir "$OUT"
git -C "$ROOT" diff --exit-code -- "data/calibration/respondent_spread/*.yaml"
```

With `--offline`, a resolver recomputes its result from the stored files and keeps the stored retrieval time. The final `git diff` reports no difference; one means that the archive, the catalogue or a script is not the committed vintage's, and must not be committed as a reassembly.

### Retrieving a new vintage

A new retrieval makes a new package with its own `retrieval_date`. It cannot be assembled into the committed vintage: the assembler stops at a seed whose source was retrieved on another UTC date than the catalogue's. Do not edit a `retrieved_at`, a result file or the system clock, because the recorded date identifies the source state behind each value. Before retrieving:

- Set the catalogue's `retrieval_date` to the UTC date on which the resolvers will run, and change it in every seed's `quantity` text, which the assembler does not check.
- To move the windows, edit each resolver's seed table, the catalogue's seed ids, `quantity` texts and `latest_training_cutoff`, and the primary-window bound in `assemble.py`. The assembler stops if a `quantity` text does not state the window its resolver reports, or if a primary or secondary window starts on or before the training cutoff.
- Change the requester identity, `AIUC calibration-seed resolver (research)`, which is fixed in eighteen resolver scripts, to identify yourself.

Then retrieve into a new empty data root and a new results directory, on the catalogue's `retrieval_date` in UTC, with outbound HTTPS:

```sh
ROOT=/path/to/cat-model
DATA=/path/to/new-empty-data
OUT=/path/to/new-empty-results
PY="$ROOT/.venv/bin/python"
export SEC_EDGAR_USER_AGENT="<organization> <operator contact>"

for source in cisa_kev nvd cisa_csaf ghsa osv exploitdb defihacklabs hhs_ocr ca_ag wa_ag \
              sec_edgar sec_litigation cfpb doj ofac federal_register nhtsa_sgo \
              nhtsa_defects openfda arxiv; do
  "$PY" -I "$ROOT/scripts/calibration_seeds/$source.py" \
    --data-dir "$DATA" --out "$OUT/$source.json"
done

"$PY" -I "$ROOT/scripts/calibration_seeds/assemble.py" --results-dir "$OUT"
```

The assembler replaces the three committed files and records the digest of each resolver as edited. Keep the data root and the results directory, which reassembling the new vintage needs, and update this README's roster, dates and reporting lags. Collect again from the new `seeds.yaml` before fitting with the new `resolutions.yaml`: the fit refuses an earlier capture when a resolver's bytes or the seed ids have changed, but not when only the catalogue's wording or the recorded values have.

## Known weaknesses

- **Clustering and bulk events.** Malicious-package records, bulk CVE publication, one breach reported to several regulators and complaint campaigns can produce heavy-tailed counts. The fit measures each seed's count clustering (NB2 phi) from its own pre-window history; trend misfit and finite-history estimation error remain.
- **Cross-block drivers.** Blocks group direct record overlap, mirrored databases and incidents shared across regulators, but broader drivers such as vulnerability discovery, breach activity and federal agency operations cross blocks. The fit reports a bootstrap range that resamples the six super-blocks of `seeds.yaml`.
- **Reporting lags.** NHTSA SGO data in this snapshot run only to mid-September 2026, so its primary window stops in August. DeFiHackLabs entries depend on contributor reproduction, and CFPB complaints can arrive up to 15 days after receipt.
- **Selection was not blind.** Some window values were visible during source checks, including KEV, HHS and NHTSA recall counts. `seeds.yaml` lists the curation-sensitive DeFiHackLabs, Exploit Database and OSV `MAL-` seeds, and the fit reports `k` and its range without them.
- **Curation-sensitive definitions and small counts.** Keyword, tag, component-code and full-text matches depend on source wording or indexing. SEC Item 1.05 filings, ICS medical advisories, malicious crates, ODI investigations and Microsoft KEV additions have low counts, and several histories contain zeros.

## Licenses and terms

`sources.yaml` records each source's terms, terms URL and automated-access basis, which the scripts follow. Attribution is required for GitHub Advisory Database data (CC BY 4.0) and the Go vulnerability database via OSV (CC BY 4.0). NVD requests the statement: “This product uses data from the NVD API but is not endorsed or certified by the NVD.” The repository's [NOTICE](../../../NOTICE) carries these attributions and the statement. CISA CSAF has no repository license file; HHS OCR, Washington's dataset metadata and OFAC's list service state no license; Exploit Database's repository has no license file at its root. Check `sources.yaml` and the linked terms before each new collection.
