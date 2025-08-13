# Branch Protection Setup Guide

This document provides instructions for repository maintainers to configure branch protection rules that enforce the development guidelines.

## 🔒 Required Branch Protection Settings

### For `main` branch:

1. **Go to**: Repository Settings → Branches → Add rule
2. **Branch name pattern**: `main`
3. **Configure the following**:

#### Protect matching branches
- ✅ Require a pull request before merging
  - ✅ Require approvals: **2**
  - ✅ Dismiss stale PR approvals when new commits are pushed
  - ✅ Require review from code owners (if CODEOWNERS file exists)
  - ✅ Restrict pushes that create files that change the code owner

#### Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ **Required status checks**:
  - `test (3.9)` - Python 3.9 tests
  - `test (3.10)` - Python 3.10 tests
  - `test (3.11)` - Python 3.11 tests
  - `test (3.12)` - Python 3.12 tests

#### Additional restrictions
- ✅ Restrict pushes to matching branches
- ✅ Allow force pushes: **NO**
- ✅ Allow deletions: **NO**

### For `develop` branch:

1. **Go to**: Repository Settings → Branches → Add rule
2. **Branch name pattern**: `develop`
3. **Configure the following**:

#### Protect matching branches
- ✅ Require a pull request before merging
  - ✅ Require approvals: **1**
  - ✅ Dismiss stale PR approvals when new commits are pushed
  - ✅ Require review from code owners (if CODEOWNERS file exists)

#### Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ **Required status checks**:
  - `test (3.9)` - Python 3.9 tests
  - `test (3.10)` - Python 3.10 tests
  - `test (3.11)` - Python 3.11 tests
  - `test (3.12)` - Python 3.12 tests

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
  --field required_status_checks='{"strict":true,"contexts":["test (3.9)","test (3.10)","test (3.11)","test (3.12)"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":2,"dismiss_stale_reviews":true}' \
  --field restrictions=null

# Protect develop branch
gh api repos/:owner/:repo/branches/develop/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["test (3.9)","test (3.10)","test (3.11)","test (3.12)"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
  --field restrictions=null
```

## 📋 Checklist for Repository Setup

- [ ] Branch protection rules configured for `main`
- [ ] Branch protection rules configured for `develop`
- [ ] CI workflow is working and reporting status checks
- [ ] Test PR created and verified protection works
- [ ] Default branch set to `develop` for new PRs
- [ ] CODEOWNERS file created (optional but recommended)
- [ ] Repository settings reviewed and secured
