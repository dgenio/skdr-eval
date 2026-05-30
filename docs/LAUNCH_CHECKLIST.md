# Public Launch Readiness Checklist

> **Purpose.** `skdr-eval` should not be broadly promoted (LinkedIn, Medium,
> Hacker News, Reddit, newsletters, conference talks) until the first-user path
> is **trustworthy, explainable, and reproducible**. Premature promotion
> generates stars without adoption — or, worse, erodes trust when the first
> copied example produces alarming output.
>
> This checklist defines what must be true before a public push. Items link to
> the issue or artifact that tracks the work. Tick an item only when it is
> genuinely done on `main`.

**Legend:** `[x]` done · `[ ]` pending · 🔗 tracking issue / artifact.

---

## 1. First impression

- [x] README hero explains the job-to-be-done in plain language. 🔗 [`README.md`](https://github.com/dgenio/skdr-eval/blob/main/README.md) — "What is this?" / "When should I use this?" / "First 10 minutes".
- [x] Install command works from a clean environment. 🔗 [`README.md` → Installation](https://github.com/dgenio/skdr-eval/blob/main/README.md#installation); CI `examples-smoke` job installs default extras and runs the quickstart.
- [x] First notebook runs (Colab-friendly). 🔗 [`examples/notebooks/01_quickstart.ipynb`](https://github.com/dgenio/skdr-eval/blob/main/examples/notebooks/01_quickstart.ipynb); executed in CI via `nbmake`.
- [x] First demo has at least one **healthy** support case. 🔗 [`docs/recipes/logs-to-experiment-card.md`](recipes/logs-to-experiment-card.md) and [`examples/notebooks/06_good_vs_bad_support.ipynb`](https://github.com/dgenio/skdr-eval/blob/main/examples/notebooks/06_good_vs_bad_support.ipynb) — the healthy path reports `support_health=ok`. (Unblocked by the #106 fix; see §2.)

## 2. Trust

- [x] #106 resolved or explicitly handled — different candidate models no longer collapse to an identical `V_hat`, and the library's own synthetic demos can report `support_health=ok` on well-overlapped data. 🔗 #106.
- [x] Report interpretation guide exists. 🔗 [`docs/report-interpretation.md`](report-interpretation.md).
- [x] Metrics glossary exists. 🔗 [`docs/metrics-glossary.md`](metrics-glossary.md).
- [x] Known limitations / when-not-to-use are visible. 🔗 [`docs/concepts/estimands-and-assumptions.md`](concepts/estimands-and-assumptions.md), [`examples/known_failures/`](https://github.com/dgenio/skdr-eval/blob/main/examples/known_failures), and the "good vs bad support" tutorial (§1).
- [ ] Claims audit complete — every quantitative claim in README/docs is backed by a test, simulation, or benchmark. 🔗 partially covered by [`docs/statistical-validation-matrix.md`](statistical-validation-matrix.md); finish before launch.

## 3. Credibility

- [ ] Benchmark page or initial benchmark result exists. 🔗 #94 (OPE benchmark harness).
- [x] Comparison page vs OBP / SCOPE-RL / d3rlpy exists. 🔗 [`docs/comparisons.md`](comparisons.md).
- [ ] Citation / DOI status clear. 🔗 [`CITATION.cff`](https://github.com/dgenio/skdr-eval/blob/main/CITATION.cff) + [`docs/zenodo.md`](zenodo.md); minting the Zenodo DOI is tracked by #77.

## 4. Contribution readiness

- [ ] Good-first issues exist and are labelled. 🔗 #76.
- [x] Contributor guide / development setup is clear. 🔗 [`CONTRIBUTING.md`](https://github.com/dgenio/skdr-eval/blob/main/CONTRIBUTING.md), [`docs/DEVELOPMENT.md`](DEVELOPMENT.md).
- [x] Issue templates exist. 🔗 [`.github/ISSUE_TEMPLATE/`](https://github.com/dgenio/skdr-eval/blob/main/.github/ISSUE_TEMPLATE).
- [ ] Public roadmap is surfaced from README and the docs site. 🔗 #76; docs site tracked by #68.

## 5. Launch assets

- [ ] At least one README visual or architecture diagram.
- [ ] One LinkedIn post draft.
- [ ] One Medium / article outline.
- [ ] One short demo GIF or screenshots of the report / stakeholder card.
- [ ] Clear call to action: try the notebook, star the repo, open issues, suggest use cases.

---

## How to use this checklist

1. **Gate, don't decorate.** Do not begin broad promotion until every item in
   §1 (First impression) and §2 (Trust) is ticked. §3–§5 strengthen the launch
   but are not hard blockers for a soft announcement.
2. **Keep it honest.** If an item regresses (e.g. a demo starts producing
   `high_risk` again), untick it and open/reference an issue.
3. **Link, don't duplicate.** Each item should point at the artifact that
   proves it, so a reviewer can verify in one click.

The published roadmap (#76) and docs site (#68) link back to this checklist so
the community can see what "ready" means.
