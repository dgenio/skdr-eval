# Error Prevention Strategy

This document outlines how we prevent CI failures and ensure code quality gates work effectively.

## 🚨 Recent Issues and Solutions

### Issue 1: Import Order Error (PLC0415)
**Problem**: `from sklearn.ensemble import RandomForestRegressor` inside test function
**Root Cause**: Import inside function instead of at module level
**Solution**: Move all imports to top of file
**Prevention**: Enhanced validation script now checks each directory individually

### Issue 2: Formatting Errors
**Problem**: `ruff format --check` failing on `src/skdr_eval/core.py` and `pairwise.py`
**Root Cause**: Files not formatted according to project standards
**Solution**: Run `ruff format` to fix formatting
**Prevention**: Validation script now mirrors CI behavior exactly

## 🛡️ Prevention Mechanisms

### 1. Enhanced Validation Script
The `scripts/validate_contribution.py` now:
- ✅ Checks each directory individually (matches CI exactly)
- ✅ Uses same commands as CI pipeline
- ✅ Provides specific fix commands for each error type
- ✅ Fails fast on first error to prevent multiple issues

### 2. Pre-commit Hooks
```yaml
# .pre-commit-config.yaml ensures:
- ruff linting and formatting
- mypy type checking
- trailing whitespace removal
- end-of-file fixing
```

### 3. Makefile Integration
```makefile
validate: ## Run comprehensive validation (matches CI exactly)
	python scripts/validate_contribution.py

format: ## Format code
	ruff format src/ tests/ examples/

lint: ## Lint code
	ruff check src/ tests/ examples/

check: lint format test ## Run all quality checks
```

### 4. CI Pipeline Mirroring
Our validation script now runs **exactly the same commands** as CI:
- `ruff check src/` `ruff check tests/` `ruff check examples/`
- `ruff format --check src/` `ruff format --check tests/` `ruff format --check examples/`

## 📋 Developer Workflow

### Before Every Commit:
1. **Run validation**: `make validate` or `python scripts/validate_contribution.py`
2. **Fix any issues**: Script provides exact fix commands
3. **Commit only when validation passes**

### Before Every Push:
1. **Validation must pass**: No exceptions
2. **Pre-commit hooks must pass**: Automatic formatting applied
3. **All tests must pass**: `make test`

### Before Creating PR:
1. **Full validation suite**: `make check`
2. **Documentation updated**: If adding features
3. **Changelog updated**: For user-facing changes

## 🔧 Specific Error Prevention

### Import Errors (PLC0415):
- **Rule**: All imports at top of file
- **Check**: Ruff linting catches this automatically
- **Fix**: Move imports to module level

### Formatting Errors:
- **Rule**: All code must pass `ruff format --check`
- **Check**: Validation script runs exact CI command
- **Fix**: Run `ruff format <directory>`

### Type Errors:
- **Rule**: All code must pass mypy
- **Check**: Validation script runs mypy on source code
- **Fix**: Add type annotations or ignore comments

## 🎯 Success Metrics

### Validation Script Effectiveness:
- ✅ **Must catch all CI-preventable errors**
- ✅ **Must provide exact fix commands**
- ✅ **Must mirror CI behavior exactly**
- ✅ **Must fail fast on first error**

### Developer Experience:
- ✅ **Clear error messages with fix instructions**
- ✅ **Fast feedback loop (< 30 seconds)**
- ✅ **Consistent between local and CI**
- ✅ **Integrated with existing workflow**

## 🚀 Continuous Improvement

### When New CI Failures Occur:
1. **Analyze**: Why didn't validation catch it?
2. **Enhance**: Update validation script to catch the issue
3. **Test**: Verify validation now catches the error
4. **Document**: Update this prevention strategy

### Regular Audits:
- **Monthly**: Review CI failure patterns
- **Quarterly**: Update validation script with new checks
- **Per Release**: Verify all prevention mechanisms working

## 📚 Related Documentation

- `DEVELOPMENT.md`: Comprehensive development guidelines
- `CONTRIBUTING.md`: Contribution workflow
- `scripts/validate_contribution.py`: Automated validation tool
- `.pre-commit-config.yaml`: Pre-commit hook configuration

---

**Remember**: The goal is **zero CI failures** due to preventable errors. Every CI failure should trigger an improvement to our prevention mechanisms.
