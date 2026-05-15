# Branch Protection Setup Guide

This document provides instructions for repository maintainers to configure branch protection rules that enforce the development guidelines.

We use **GitHub Flow**: short-lived feature branches off `main`, merged back via PR.

## 🔒 Required Branch Protection Settings

### For `main` branch:

1. **Go to**: Repository Settings → Branches → Add rule
2. **Branch name pattern**: `main`
3. **Configure the following**:

#### Protect matching branches
- ✅ Require a pull request before merging
  - ✅ Require approvals: **1**
  - ✅ Dismiss stale PR approvals when new commits are pushed
  - ✅ Require review from code owners (if CODEOWNERS file exists)

#### Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ **Required status checks**:
  - `test (3.11)` - Python 3.11 tests
  - `test (3.12)` - Python 3.12 tests
  - `test (3.13)` - Python 3.13 tests
  - `test (3.14)` - Python 3.14 tests
  - `examples-smoke` - quickstart/preflight examples
  - `choice-extra-smoke` - smoke under `[choice]` extra
  - `build` - package build + twine check

#### Additional restrictions
- ✅ Restrict pushes to matching branches
- ✅ Allow force pushes: **NO**
- ✅ Allow deletions: **NO**

## 🎯 Verification

After setting up branch protection, verify by:

1. **Test PR creation**: Create a test feature branch and PR
2. **Verify CI requirement**: Ensure PR cannot be merged without CI pass
3. **Test approval requirement**: Ensure required approvals are enforced
4. **Test direct push**: Verify direct pushes to protected branches are blocked

## 🔧 GitHub CLI Setup (Alternative)

```bash
# Protect main branch
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["test (3.11)","test (3.12)","test (3.13)","test (3.14)","examples-smoke","choice-extra-smoke","build"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
  --field restrictions=null
```

## 📋 Checklist for Repository Setup

- [ ] Branch protection rules configured for `main`
- [ ] CI workflow is working and reporting status checks
- [ ] Test PR created and verified protection works
- [ ] Default branch set to `main`
- [ ] CODEOWNERS file created (optional but recommended)
- [ ] Repository settings reviewed and secured
