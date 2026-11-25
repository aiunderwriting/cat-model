# External reference data

This Git-ignored directory can hold public or licensed reference snapshots that an operator selects. No command downloads, fits or reads them. A study that uses a file captures it in its source bundle with its provenance ([runbook section 1](../../runbook.md#1-evidence-and-scope)).

## Source inventory and applicability

| Source | Conventional local location | What it can and cannot show |
|---|---|---|
| [CISA Known Exploited Vulnerabilities](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) | `cisa_kev/known_exploited_vulnerabilities.json` | Exploited-vulnerability context. Entries are neither campaigns nor matched frontier-AI loss occurrences. |
| [NHTSA Standing General Order reports](https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting) | `nhtsa/` | Autonomous-system incident context. Several reports or report versions can describe one incident, and without exposure the reports give no accident rate. |
| [NPDB public-use file](https://www.npdb.hrsa.gov/resources/publicData.jsp) | `npdb/` | Payer-practitioner payment reports and their reporting and payment history. They do not identify unique patients, complete gross loss or AI clinical-decision exposure. |
| [FRI XPT](https://github.com/forecastingresearch/xpt-lib) | `fri/xpt-lib/` | Long-horizon forecaster disagreement and correlated beliefs. It does not identify annual loss dependence conditional on ECI. |
| [AI Incident Database](https://incidentdatabase.ai/) | `aiid/incidents.csv` | Reported incidents and harm types, with selection and reporting limitations; no standardized insured severity population. |

Keep original records in their source units. An NPDB payment-band analysis keeps the intervals, the public top code and the inflation basis, and does not treat unknown policy limits as censoring. A revised crash report keeps the version available at the study cutoff. Missing or zero amounts need explicit treatment, not silent exclusion or invented values.

[Analog evidence](../../references/calibration/README.md) summarizes results from some of these sources. Other sources, such as Schedule P, GTD/START or SEC filings, follow the same capture rules.

## Access and redistribution terms

None of these files is committed. A study that captures one in a source bundle takes on its terms, so check a provider's current terms before sharing source bytes. The terms below are as read on 2026-11-11.

| Source | Terms | Consequence for a source bundle |
|---|---|---|
| CISA Known Exploited Vulnerabilities | [CC0 1.0](https://github.com/cisagov/kev-data), as CISA's mirror of the catalog states. | The file may be captured and shared. |
| NHTSA Standing General Order reports | NHTSA's [terms of use](https://www.nhtsa.gov/about-nhtsa/terms-use) treat information presented on its website as public information that may be distributed or copied. | The files may be captured and shared. |
| NPDB public-use file | HRSA [data use agreement](https://www.npdb.hrsa.gov/resources/publicData.jsp): no identification of any individual or entity, and anyone who reposts the file, or a subset of it, must bind its users to the same conditions. | Never commit or publish its bytes, including in a shared source bundle. Aggregate results that identify no one, such as the [NPDB tail fit](../../references/calibration/npdb_medical_severity.md), may be reported. |
| FRI XPT | [MIT License](https://github.com/forecastingresearch/xpt-lib/blob/main/LICENSE.md) for the replication package. | The files may be captured and shared with their copyright and permission notice. |
| AI Incident Database | [CC BY-SA 4.0](https://incidentdatabase.ai/terms-of-use/) for the snapshot collections `incidents`, `quickadd`, `duplicates`, `taxa`, `classifications`, `entities` and `entity_relationships`. The text of reports, submissions and linked media are excluded from that licence. | The snapshot collections may be captured and shared with credit under the same licence; report text may not. |
