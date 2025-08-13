# Development Guidelines for skdr-eval

> **🤖 AI Agent Optimized**: This guide is designed for both human developers and AI coding assistants to ensure consistent, high-quality contributions.

## 🎯 Quick Start for AI Agents

### Essential Commands (Always Use These)
```bash
# 1. Setup development environment
make install-dev

# 2. Run all quality checks before any commit
make check

# 3. Format code automatically
make format

# 4. Run tests with coverage
make test-cov
```

### Critical Rules (Never Break These)
1. **NEVER commit directly to `main` or `develop`**
2. **ALWAYS create feature branches from `develop`**
3. **ALWAYS run `make check` before creating PR**
4. **ALWAYS write tests for new functionality**
5. **ALWAYS update documentation for API changes**

## 🌊 Git Flow Workflow (SOTA Implementation)

### Branch Strategy
```
main (production)     ←── release/v1.2.0 ←── hotfix/critical-bug
  ↑                                              ↑
  └── develop (integration) ←── feature/new-algo ←── feature/bug-fix
```

### Step-by-Step Development Process

#### 1. Start New Feature
```bash
# Always start from develop
git checkout develop
git pull origin develop
git checkout -b feature/descriptive-name

# Example naming conventions:
# feature/add-cosine-similarity
# feature/fix-memory-leak
# feature/improve-performance
# hotfix/critical-security-fix
```

#### 2. Development Loop
```bash
# Make changes, then run quality checks
make check                    # Runs lint + typecheck + test
make format                   # Auto-format code

# Commit with conventional format
git add .
git commit -m "feat: add cosine similarity metric

- Implement cosine similarity calculation
- Add comprehensive tests with edge cases
- Update documentation with examples
- Benchmark shows 15% performance improvement

Closes #123"
```

#### 3. Pre-PR Validation
```bash
# MANDATORY: Run full validation suite
make clean                    # Clean artifacts
make check                    # All quality checks
make test-cov                 # Tests with coverage report

# Check coverage is ≥80%
# Verify all tests pass
# Ensure no linting errors
```

#### 4. Create Pull Request
```bash
git push origin feature/descriptive-name

# Create PR with template:
# - Target: develop branch
# - Fill ALL template sections
# - Link related issues
# - Add screenshots/examples if UI changes
```

## 🔍 Code Quality Standards (Enforced by CI)

### Linting & Formatting
- **Tool**: Ruff (replaces flake8, isort, black)
- **Line length**: 88 characters
- **Import sorting**: Automatic with ruff
- **Command**: `make lint` and `make format`

### Type Checking
- **Tool**: mypy with strict configuration
- **Requirement**: All functions MUST have type hints
- **Command**: `make typecheck`
- **Config**: `mypy.ini`

### Testing Requirements
- **Framework**: pytest
- **Coverage**: Minimum 80% (enforced by CI)
- **Types**: Unit, integration, property-based tests
- **Command**: `make test-cov`

### Documentation Standards
- **Docstrings**: Google-style for all public APIs
- **Type hints**: Required for all functions
- **Examples**: Include usage examples in docstrings
- **API changes**: Must update README.md

## 🤖 AI Agent Specific Guidelines

### Code Analysis Checklist
Before making any changes, AI agents should:

1. **Understand Context**
   ```bash
   # Examine the codebase structure
   find src/ -name "*.py" | head -10

   # Check existing tests
   find tests/ -name "*.py" | head -5

   # Review recent changes
   git log --oneline -10
   ```

2. **Identify Dependencies**
   ```bash
   # Check what imports the module you're changing
   grep -r "from skdr_eval.your_module import" src/

   # Check what your module imports
   grep "^import\|^from" src/skdr_eval/your_module.py
   ```

3. **Validate Changes**
   ```bash
   # Test specific module
   pytest tests/test_your_module.py -v

   # Test related functionality
   pytest -k "your_feature" -v

   # Full validation
   make check
   ```

### Common Patterns for AI Agents

#### Adding New Functionality
```python
# 1. Always add type hints
def new_function(data: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """Brief description of what the function does.

    Args:
        data: Description of the input parameter.

    Returns:
        Tuple containing result and metadata.

    Raises:
        ValueError: When input data is invalid.

    Example:
        >>> result, meta = new_function([{"key": "value"}])
        >>> print(result)
        0.85
    """
    # Implementation here
    pass

# 2. Always add corresponding tests
def test_new_function():
    """Test the new_function with various inputs."""
    # Test normal case
    result, meta = new_function([{"key": "value"}])
    assert isinstance(result, float)

    # Test edge cases
    with pytest.raises(ValueError):
        new_function([])
```

#### Modifying Existing Code
```python
# 1. Check existing tests first
# 2. Understand the current behavior
# 3. Make minimal changes
# 4. Add tests for new behavior
# 5. Ensure backward compatibility
```

### Error Handling Patterns
```python
# Use specific exceptions
raise ValueError(f"Invalid input: {input_value}")

# Log important information
import logging
logger = logging.getLogger(__name__)
logger.info(f"Processing {len(data)} items")

# Handle edge cases explicitly
if not data:
    raise ValueError("Input data cannot be empty")
```

## 🚀 CI/CD Pipeline (GitHub Actions)

### Automated Checks (Must Pass)
1. **Linting**: `ruff check` on Python 3.9-3.12
2. **Formatting**: `ruff format --check`
3. **Type checking**: `mypy src/skdr_eval/`
4. **Testing**: `pytest` with coverage report
5. **Coverage**: Minimum 80% required

### PR Requirements
- ✅ All CI checks pass
- ✅ At least 1 approval from maintainer
- ✅ Branch is up-to-date with develop
- ✅ No merge conflicts
- ✅ PR template fully completed

### Branch Protection Rules
- **main**: Requires PR + 2 approvals + CI pass
- **develop**: Requires PR + 1 approval + CI pass
- **feature/***: No restrictions (for development)

## 📦 Release Process (Semantic Versioning)

### Version Types
- **MAJOR** (1.0.0 → 2.0.0): Breaking changes
- **MINOR** (1.0.0 → 1.1.0): New features (backward compatible)
- **PATCH** (1.0.0 → 1.0.1): Bug fixes

### Release Steps (Maintainers Only)
```bash
# 1. Create release branch
git checkout develop
git checkout -b release/v1.2.0

# 2. Update version and changelog
# Edit pyproject.toml, CHANGELOG.md

# 3. Create PR to main
# After approval and merge:

# 4. Tag and publish
git checkout main
git tag v1.2.0
git push origin v1.2.0

# 5. Merge back to develop
git checkout develop
git merge main
```

## 🛠️ Development Environment Setup

### Prerequisites
- Python 3.9+ (tested on 3.9, 3.10, 3.11, 3.12)
- Git
- GitHub account with SSH keys configured

### Local Setup
```bash
# 1. Clone and setup
git clone git@github.com:dgenio/skdr-eval.git
cd skdr-eval

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install in development mode
make install-dev

# 4. Setup pre-commit hooks (recommended)
pre-commit install

# 5. Verify setup
make check
```

### IDE Configuration

#### VS Code Settings
```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false
}
```

#### PyCharm Settings
- Enable ruff for linting and formatting
- Set line length to 88
- Enable pytest as test runner
- Configure mypy as external tool

## 🔧 Troubleshooting Common Issues

### CI Failures
```bash
# Linting errors
make lint                     # See errors
make format                   # Auto-fix formatting

# Type checking errors
make typecheck               # See mypy errors
# Fix type hints manually

# Test failures
make test-cov                # Run with coverage
pytest tests/test_file.py -v # Run specific test
pytest -k "test_name" -v     # Run specific test by name

# Coverage too low
pytest --cov-report=html     # Generate HTML report
open htmlcov/index.html      # See what's missing
```

### Git Issues
```bash
# Merge conflicts
git status                   # See conflicted files
# Resolve conflicts manually
git add .
git commit

# Branch out of sync
git checkout develop
git pull origin develop
git checkout feature/branch
git rebase develop          # Or merge develop
```

## 📊 Performance Guidelines

### Benchmarking
```python
import time
import cProfile

# Time critical functions
start = time.perf_counter()
result = your_function(data)
duration = time.perf_counter() - start
print(f"Function took {duration:.4f} seconds")

# Profile complex operations
cProfile.run('your_function(data)')
```

### Memory Usage
```python
import tracemalloc

# Monitor memory usage
tracemalloc.start()
result = your_function(data)
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.2f} MB")
print(f"Peak: {peak / 1024 / 1024:.2f} MB")
```

## 🎯 AI Agent Success Checklist

Before submitting any PR, ensure:

- [ ] **Environment**: `make install-dev` completed successfully
- [ ] **Quality**: `make check` passes without errors
- [ ] **Tests**: New functionality has comprehensive tests
- [ ] **Coverage**: Overall coverage remains ≥80%
- [ ] **Documentation**: Public APIs have proper docstrings
- [ ] **Types**: All functions have type hints
- [ ] **Commits**: Follow conventional commit format
- [ ] **Branch**: Created from `develop`, targets `develop`
- [ ] **PR**: Template fully completed with clear description

## 🤝 Getting Help

- **Issues**: Create GitHub issue with bug/feature template
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Tag maintainers in PR for review
- **Documentation**: Check README.md and docstrings first

---

> **Remember**: This guide ensures consistent, high-quality contributions. Following these guidelines helps maintain code quality and makes collaboration smooth for both humans and AI agents.
