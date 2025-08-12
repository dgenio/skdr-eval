# skdr-eval v0.1.1 Release Notes

🔧 **Quality and Infrastructure Improvements**

This patch release focuses on code quality improvements, linting fixes, and GitHub Actions workflow updates to ensure a smooth development and release process.

## 🛠️ What's Fixed

### 🧹 **Code Quality & Linting**
- **Fixed ruff configuration**: Resolved `.ruff.toml` configuration error that prevented linting from running
- **Comprehensive linting cleanup**: Fixed 257+ linting issues across the codebase
- **Modern type annotations**: Updated from `Dict`/`Tuple` to `dict`/`tuple` syntax
- **Code style improvements**: Applied automated formatting and style fixes
- **All checks pass**: Codebase now passes all ruff quality checks

### 🚀 **GitHub Actions & CI/CD**
- **Fixed workflow deprecation warnings**: Updated `actions/upload-artifact` from v3 to v4
- **Updated Python setup**: Updated `actions/setup-python` from v4 to v5
- **Reliable releases**: GitHub Actions workflows now run without errors
- **Improved CI reliability**: Both CI and release workflows updated and tested

### 📝 **Documentation & Release Process**
- **Comprehensive release notes**: Added detailed v0.1.0 release notes
- **Better commit messages**: Improved commit history with descriptive messages
- **Clean repository**: All changes properly committed and organized

## 🔍 Technical Details

### Linting Improvements
- Fixed 206 issues automatically with `ruff --fix`
- Fixed 46 additional issues with `ruff --fix --unsafe-fixes`
- Manually resolved remaining 5 issues with appropriate `noqa` comments
- Updated imports and removed unused variables

### Configuration Fixes
- Moved `line-length = 88` from `[lint]` section to top-level in `.ruff.toml`
- This resolved the critical configuration error preventing ruff from running

### Workflow Updates
- **actions/setup-python@v5**: Latest stable version for Python setup
- **actions/upload-artifact@v4**: Resolves deprecation warnings
- Applied to both CI and release workflows for consistency

## 📦 Installation

```bash
pip install skdr-eval==0.1.1
```

Or upgrade from previous version:
```bash
pip install --upgrade skdr-eval
```

## 🔄 Migration from v0.1.0

No breaking changes! This is a drop-in replacement for v0.1.0 with improved code quality and infrastructure.

All existing code using v0.1.0 will work unchanged with v0.1.1.

## 🎯 What's Next

With the infrastructure and code quality improvements in place, future releases will focus on:
- New estimator methods
- Enhanced visualization capabilities
- Performance optimizations
- Additional synthetic data generators

## 🤝 Contributing

The improved linting setup makes contributing easier:
- Run `ruff check src/ tests/ examples/` to check code quality
- Run `ruff check --fix src/ tests/ examples/` to auto-fix issues
- All CI checks now run reliably

## 📄 License

MIT License - see LICENSE file for details.

---

**Upgrade today** with `pip install --upgrade skdr-eval` to get the latest quality improvements and ensure compatibility with modern Python tooling!

For questions or issues, visit our [GitHub repository](https://github.com/dgenio/skdr-eval).
