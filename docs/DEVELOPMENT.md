# Development Workflow Guide

This document outlines the professional development workflow for skdr-eval, designed to maintain high code quality and enable collaborative development.

## 🌟 Branch Strategy

We follow a **Git Flow** inspired workflow:

```
main (production)     ←─── hotfix/critical-fix
  ↑                         ↗
  └─── release/v1.2.0 ←─── develop (integration)
                             ↑
                             ├─── feature/new-estimator
                             ├─── feature/improved-docs
                             └─── feature/performance-boost
```

### Branch Types

| Branch | Purpose | Lifetime | Protected |
|--------|---------|----------|-----------|
| `main` | Production releases | Permanent | ✅ |
| `develop` | Integration branch | Permanent | ✅ |
| `feature/*` | New features | Temporary | ❌ |
| `hotfix/*` | Critical fixes | Temporary | ❌ |
| `release/*` | Release preparation | Temporary | ❌ |

## 🔄 Development Workflow

### 1. Starting New Work

```bash
# For new features
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# For hotfixes (rare)
git checkout main
git pull origin main
git checkout -b hotfix/critical-issue
```

### 2. Development Process

```bash
# Make your changes
# Run quality checks frequently
make check

# Commit with conventional commit messages
git add .
git commit -m "feat(core): add MAGIC estimator implementation"
```

### 3. Pre-submission Checklist

- [ ] Code follows style guidelines (`make format`)
- [ ] All tests pass (`make test`)
- [ ] Type checking passes (`make typecheck`)
- [ ] Linting passes (`make lint`)
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated for significant changes

### 4. Pull Request Process

```bash
# Push feature branch
git push -u origin feature/your-feature-name

# Create PR via GitHub UI
# - Target: develop branch (not main!)
# - Fill out PR template completely
# - Link related issues
```

### 5. Code Review & Merge

- Automated CI checks must pass
- At least one maintainer review required
- Address feedback promptly
- Squash commits if requested
- Maintainer merges when approved

## 🛠️ Development Tools

### Essential Commands

```bash
# Setup development environment
make install-dev

# Run all quality checks
make check

# Individual checks
make lint      # Ruff linting
make format    # Code formatting
make typecheck # mypy type checking
make test      # pytest with coverage

# Build for release
make build

# Clean artifacts
make clean
```

### Pre-commit Hooks (Recommended)

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 📋 Code Quality Standards

### Linting & Formatting
- **Tool**: Ruff (replaces black, isort, flake8)
- **Line length**: 88 characters
- **Style**: PEP 8 compliant
- **Import sorting**: Automatic with ruff

### Type Checking
- **Tool**: mypy
- **Coverage**: All public APIs must be typed
- **Style**: Modern type hints (`dict` not `Dict`)

### Testing
- **Framework**: pytest
- **Coverage**: Minimum 80% line coverage
- **Types**: Unit, integration, and smoke tests
- **Naming**: `test_*.py` files in `tests/` directory

### Documentation
- **Docstrings**: Google-style for all public APIs
- **Examples**: Working code examples in docstrings
- **README**: Keep installation and usage up-to-date
- **CHANGELOG**: Document all user-facing changes

## 🚀 Release Process

### Version Numbering
Follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Create Release Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b release/v1.2.0
   ```

2. **Prepare Release**
   - Update version in relevant files
   - Update CHANGELOG.md
   - Run full test suite
   - Update documentation

3. **Release PR**
   ```bash
   git push -u origin release/v1.2.0
   # Create PR: release/v1.2.0 → main
   ```

4. **After Merge**
   ```bash
   # Tag the release
   git checkout main
   git pull origin main
   git tag v1.2.0
   git push origin v1.2.0

   # Merge back to develop
   git checkout develop
   git merge main
   git push origin develop
   ```

5. **Automated Publishing**
   - GitHub Actions automatically publishes to PyPI
   - Creates GitHub release with notes

## 🔍 Code Review Guidelines

### For Authors
- **Small PRs**: Keep changes focused and reviewable
- **Clear descriptions**: Explain what and why
- **Self-review**: Review your own PR first
- **Tests included**: New features need tests
- **Documentation**: Update docs for API changes

### For Reviewers
- **Be constructive**: Suggest improvements
- **Check functionality**: Does it work as intended?
- **Verify tests**: Are edge cases covered?
- **Consider maintainability**: Is code readable?
- **Performance**: Any performance implications?

## 🧪 Testing Strategy

### Test Categories
1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **Smoke Tests**: Basic functionality verification
4. **Performance Tests**: Benchmark critical paths

### Test Organization
```
tests/
├── test_api.py              # Public API tests
├── test_core.py             # Core functionality
├── test_synth.py            # Synthetic data generation
├── test_integration.py      # Integration tests
└── test_performance.py      # Performance benchmarks
```

### Coverage Requirements
- **Minimum**: 80% line coverage
- **Target**: 90% line coverage
- **Critical paths**: 100% coverage
- **New features**: Must include tests

## 📝 Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Examples
```
feat(core): add MAGIC estimator implementation
fix(synth): handle edge case in data generation
docs(readme): update installation instructions
test(core): add integration tests for DR estimation
chore(deps): update scikit-learn to v1.3.0
```

## 🔧 Troubleshooting

### Common Issues

**Q: CI fails with linting errors**
```bash
# Fix locally
make format
make lint
git add .
git commit -m "style: fix linting issues"
```

**Q: Tests fail locally but pass in CI**
```bash
# Ensure clean environment
make clean
make install-dev
make test
```

**Q: Type checking fails**
```bash
# Run mypy locally
make typecheck
# Fix type annotations
```

**Q: Coverage too low**
```bash
# Check coverage report
make test-cov
# Add tests for uncovered lines
```

### Getting Help
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Code Review**: Ask questions in PR comments
- **Documentation**: Check README and API docs

---

This development workflow ensures high code quality, maintainable codebase, and smooth collaboration. Welcome to the skdr-eval development community! 🚀
