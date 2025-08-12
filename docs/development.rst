Development Guide
=================

This page provides information for developers who want to contribute to skdr-eval.

Development Setup
-----------------

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/dandrsantos/skdr-eval.git
      cd skdr-eval

2. **Install in development mode**:

   .. code-block:: bash

      pip install -e .[dev]

3. **Set up pre-commit hooks**:

   .. code-block:: bash

      pre-commit install

Development Workflow
--------------------

We follow a **Git Flow** inspired workflow:

Branch Structure
~~~~~~~~~~~~~~~~

- **`main`**: Production-ready code, protected branch
- **`develop`**: Integration branch for features, protected branch  
- **`feature/*`**: Feature development branches
- **`hotfix/*`**: Critical fixes for production
- **`release/*`**: Release preparation branches

Making Changes
~~~~~~~~~~~~~~

1. **Create Feature Branch**:

   .. code-block:: bash

      git checkout develop
      git pull origin develop
      git checkout -b feature/your-feature-name

2. **Development**:

   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all checks pass locally

3. **Pre-commit Checks**:

   .. code-block:: bash

      # Run all checks
      make check
      
      # Or run individually
      make lint      # Linting with ruff
      make format    # Code formatting
      make typecheck # Type checking with mypy
      make test      # Run tests with pytest

4. **Submit Pull Request**:

   - Push feature branch to origin
   - Create PR against `develop` branch
   - Fill out PR template completely
   - Wait for CI checks and code review

Code Quality Standards
----------------------

Linting and Formatting
~~~~~~~~~~~~~~~~~~~~~~~

We use **ruff** for both linting and formatting:

.. code-block:: bash

   # Check for issues
   ruff check src/ tests/ examples/
   
   # Auto-format code
   ruff format src/ tests/ examples/

Type Checking
~~~~~~~~~~~~~

We use **mypy** for static type checking:

.. code-block:: bash

   mypy src/skdr_eval/

All public functions must have type annotations.

Testing
~~~~~~~

We use **pytest** for testing:

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=skdr_eval --cov-report=html

- Aim for >90% test coverage
- Write unit tests for all new functions
- Include integration tests for complex workflows

Documentation
~~~~~~~~~~~~~

We use **Sphinx** for documentation:

.. code-block:: bash

   # Build documentation
   cd docs/
   make html
   
   # View documentation
   open _build/html/index.html

- Document all public APIs with docstrings
- Use Google-style docstrings
- Include examples in docstrings where helpful

Release Process
---------------

1. **Prepare Release**:

   .. code-block:: bash

      git checkout develop
      git pull origin develop
      git checkout -b release/vX.Y.Z

2. **Update Version and Changelog**:

   - Update version in `pyproject.toml`
   - Update `CHANGELOG.md`
   - Create release notes

3. **Final Testing**:

   .. code-block:: bash

      make check
      make build

4. **Merge to Main**:

   - Create PR from release branch to `main`
   - After approval and merge, tag the release
   - Merge `main` back to `develop`

5. **Publish**:

   .. code-block:: bash

      # Build and publish to PyPI
      make build
      twine upload dist/*

Docker Development
------------------

For consistent development environments, use Docker:

.. code-block:: bash

   # Build development image
   docker build -t skdr-eval-dev .
   
   # Run development container
   docker run -it -v $(pwd):/workspace skdr-eval-dev bash

GitHub Codespaces
-----------------

For instant development environments, use GitHub Codespaces:

1. Click "Code" → "Codespaces" → "Create codespace on develop"
2. Wait for environment to initialize
3. Start developing immediately

The `.devcontainer/` configuration provides:

- Python 3.11 environment
- All development dependencies pre-installed
- VS Code extensions for Python development
- Pre-commit hooks configured

Contributing Guidelines
-----------------------

Code Style
~~~~~~~~~~

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write clear, descriptive variable and function names
- Keep functions focused and single-purpose

Documentation
~~~~~~~~~~~~~

- Write docstrings for all public functions and classes
- Use Google-style docstring format
- Include examples in docstrings where helpful
- Update relevant documentation files

Testing
~~~~~~~

- Write tests for all new functionality
- Ensure tests are deterministic and isolated
- Use descriptive test names
- Aim for high test coverage

Pull Requests
~~~~~~~~~~~~~

- Fill out the PR template completely
- Reference related issues
- Provide clear description of changes
- Ensure all CI checks pass
- Request review from maintainers

Issue Reporting
~~~~~~~~~~~~~~~

When reporting issues:

- Use the appropriate issue template
- Provide minimal reproducible example
- Include environment information
- Be clear and specific about the problem

Getting Help
------------

- **Documentation**: Check the API reference and examples
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues
