# Release Process

A software release is a tag of the code, with the wheel and source distribution built from that tag attached to its GitHub Release. Each reported study keeps its own records, listed below, and is reproduced with the code revision they record.

## Software releases

The `pyproject.toml` version uses `MAJOR.MINOR.PATCH`. The software is experimental, and a `0.y.z` release can make breaking changes. A patch version marks corrections or changes that leave contracts alone; a minor version marks added capabilities or materially changed contracts. Release notes live in the GitHub Release of each tag.

Before tagging, update the version in `pyproject.toml` and [CITATION.cff](CITATION.cff), regenerate the lockfile and complete the [repository checks](TESTING.md#repository-checks) and CI. A test in `tests/test_release_invariants.py` fails while the two versions differ. Never move a published release tag.

The `Release artifact` workflow (`.github/workflows/release.yml`) runs in every CI run as part of `ci-green`, on pushed `X.Y.Z` or `vX.Y.Z` tags and on manual dispatch. It checks the lockfile, validates the project version and tag, builds the wheel and source distribution, rejects forbidden wheel contents and verifies required packaged resources. It runs the [source-distribution check](TESTING.md#source-distribution-check), which requires the archive to carry the tracked tree and nothing Git does not track, with every relative Markdown link resolving inside it. It then runs the [installed-wheel acceptance](TESTING.md#installed-wheel-acceptance) check, checks that the installed CLI has exactly the commands and options listed in `tests/test_cli_stages.py`, smoke-tests model profiles, scenario loading and provider transport with a fake completion, and uploads `acceptance.json` as the `installed-wheel-acceptance` workflow artifact, retained for 14 days. On a tag, once every check has passed, it also uploads the wheel and source distribution it built and checked, with a `SHA256SUMS` file of their SHA-256 digests, as the `release-distributions` workflow artifact, retained for 90 days. The workflow does not create the GitHub Release or publish to a package index.

A maintainer creates the GitHub Release and attaches those files as they are, without rebuilding them. Find the tag's `Release artifact` run on the repository's Actions page or with `gh run list --workflow release.yml`, then:

```sh
gh run download <run-id> --name release-distributions --dir release-distributions
(cd release-distributions && sha256sum --check SHA256SUMS)  # macOS: shasum -a 256 --check SHA256SUMS
gh release create <tag> --verify-tag --title <tag> --notes-file <release-notes> release-distributions/*
```

`gh release upload <tag> release-distributions/*` adds the files to a release that already exists. The wheel's digest in `SHA256SUMS` equals `wheel_sha256` in that run's `acceptance.json`, which ties the attached wheel to the acceptance it passed.

## Study records

Keep with each reported study, in access-controlled storage:

- the setup manifest, source bundle and verified local source bytes;
- the capability evidence and forecast, where used;
- any risk instrument, the declared respondent roster, every raw collection attempt and the accepted responses;
- the entity and population roster, the complete loss laws and any financial inputs;
- each assumption's rationale and evidence status, and any review records;
- the code and dependency versions, run controls, run capture, results and numerical diagnostics;
- the required comparisons, those that ran and those that did not, and the material limitations;
- the study's purpose, source licenses and access conditions.

A source record keeps the provider or source, its URL or captured path, checksum, observation, vintage, availability and retrieval dates, filters and transformations, and access or redistribution restrictions. Keep the bytes actually used; a live URL is no substitute. Backups and restore checks cover this storage as well as the code.

Records are immutable. A correction gets a new identity, and an earlier record keeps its original versions and identity, never relabeled as current.

A comparison with an earlier study names what changed between them: populations, dates, evidence, judgments, probability laws, model structure, ownership and financial terms. Use paired seeds where meaningful, keep the numerical limitations and do not attribute all movement to one change when other inputs differ.
