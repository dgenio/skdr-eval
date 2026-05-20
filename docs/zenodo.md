# Minting a Zenodo DOI

This document is the one-time, maintainer-only checklist for binding
`skdr-eval` to [Zenodo](https://zenodo.org) so every tagged GitHub release
gets archived with a citable DOI.

## Why

A non-trivial fraction of academic users will only cite — and therefore only
*use* — a library that has a versioned DOI. The DOI also gives the project a
permanent landing page independent of GitHub.

## One-time setup (≈5 minutes)

1. Sign in to <https://zenodo.org> with the GitHub identity that has admin
   rights on `dgenio/skdr-eval` (free account; no fee).
2. Open <https://zenodo.org/account/settings/github/> and click **"Sync now"**.
3. Find `dgenio/skdr-eval` in the repository list and flip the toggle to **On**.
   This authorizes Zenodo to listen for GitHub release webhooks on this repo.
4. Confirm the metadata. Zenodo will use `.zenodo.json` at the repo root
   (already present) as the source of truth for title, authors, keywords,
   description, and license.

## First release that gets a DOI

The next tagged release (after `v0.7.0`) will automatically:

1. Trigger the existing `release.yml` workflow.
2. Notify Zenodo via the GitHub webhook.
3. Archive the tagged source tree.
4. Mint **two** DOIs:
   - A *concept DOI* representing "all versions of skdr-eval".
   - A *version DOI* specific to that release.

## Post-mint follow-up

Once the concept DOI is known, replace the placeholder in two files:

- `CITATION.cff` — the `identifiers:` block (`type: doi`) currently holds
  `10.5281/zenodo.0000000`. Replace with the real concept DOI.
- `README.md` — add a Zenodo DOI badge under the existing PyPI / Coverage
  badges (suggested badge URL pattern is provided on the Zenodo record
  page).

Open a small `chore: bind real Zenodo DOI` PR with both edits.

## Methods note / preprint

A short methods note describing what `skdr-eval` does differently from the
broader OPE ecosystem (Open Bandit Pipeline, SCOPE-RL, banditml) is drafted
in [`docs/methods.md`](methods.md). After the DOI is live, that note can be
deposited on arXiv (cs.LG / stat.ML) and the arXiv ID added under
`identifiers:` in `CITATION.cff`.
