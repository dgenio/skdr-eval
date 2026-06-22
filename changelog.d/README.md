# Changelog fragments

This directory holds **unreleased** changelog entries, one file per change.
They are compiled into a dated section at the top of
[`../CHANGELOG.md`](../CHANGELOG.md) at release time. This replaces hand-editing
a shared `## [Unreleased]` block — the single most frequent merge-conflict
source in the repo (#255).

## Add a fragment in your PR

Create a file named `<issue>.<type>.md`, where `<issue>` is the GitHub issue or
PR number and `<type>` is one of:

| Type         | Use for                                                        |
|--------------|----------------------------------------------------------------|
| `added`      | New features / public API.                                     |
| `changed`    | Changes in existing behaviour (incl. **default flips**).       |
| `deprecated` | Soon-to-be-removed features.                                   |
| `removed`    | Removed features / public API.                                 |
| `fixed`      | Bug fixes.                                                     |
| `security`   | Security-relevant fixes.                                       |

The file body is the entry text (Markdown). One sentence is ideal; reference the
issue with `#123` if helpful.

```bash
# Example
echo "Vectorized the propensity hot path for a 3x speedup on large logs." \
  > changelog.d/209.changed.md
```

> **Statistical changes:** any change that flips a default on a statistical
> entry point needs a `changed` fragment (see `CONTRIBUTING.md`).

## Preview and build

```bash
make changelog-draft   # preview the compiled notes (no files changed)
make changelog         # at release time: render fragments into CHANGELOG.md
```

`make changelog` consumes the fragments (deletes them) and writes a dated
`## [X.Y.Z]` section. Run it on the release-prep commit, not in feature PRs.
