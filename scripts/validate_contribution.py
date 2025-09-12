#!/usr/bin/env python3
"""
Comprehensive contribution validation script for skdr-eval.

This script validates contributions before they are submitted as PRs,
ensuring they meet all quality standards and will pass CI checks.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Configuration constants
COVERAGE_THRESHOLD = 80
CODE_DIRS = ("src/", "tests/", "examples/")
PROTECTED_BRANCHES = ("main", "develop")
CONVENTIONAL_PREFIXES = (
    "feat:",
    "fix:",
    "docs:",
    "style:",
    "refactor:",
    "test:",
    "chore:",
)


class ContributionValidator:
    """Validates contributions against skdr-eval development standards."""

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize validator with repository root."""
        self.repo_root = repo_root or Path.cwd()
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.success_count = 0
        self.total_checks = 0

    def run_command(
        self, cmd: list[str], capture_output: bool = True
    ) -> tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=capture_output,
                text=True,
                cwd=self.repo_root,
            )
            return result.returncode, result.stdout, result.stderr
        except FileNotFoundError:
            return 1, "", f"Command not found: {cmd[0]}"

    def check_git_status(self) -> bool:
        """Check git repository status."""
        print("Checking git repository status...")
        self.total_checks += 1

        # Check if we're in a git repository
        code, _, _ = self.run_command(["git", "status", "--porcelain"])
        if code != 0:
            self.errors.append("Not in a git repository or git not available")
            return False

        # Check current branch
        code, stdout, _ = self.run_command(["git", "branch", "--show-current"])
        if code != 0:
            self.errors.append("Could not determine current branch")
            return False

        current_branch = stdout.strip()
        if current_branch in PROTECTED_BRANCHES:
            self.errors.append(
                f"You are on protected branch '{current_branch}'. Create a feature branch first!"
            )
            return False

        # Check if branch is ahead of develop
        code, stdout, _ = self.run_command(
            ["git", "rev-list", "--count", "develop..HEAD"]
        )
        if code == 0:
            commits_ahead = int(stdout.strip()) if stdout.strip().isdigit() else 0
            if commits_ahead == 0:
                self.warnings.append(
                    "No new commits on this branch compared to develop"
                )

        self.success_count += 1
        print(f"Git status OK (branch: {current_branch})")
        return True

    def check_linting(self) -> bool:
        """Check code linting with ruff - exactly as strict as CI."""
        print("Checking code linting (CI-strict)...")
        self.total_checks += 1

        # Check each directory individually to match CI behavior exactly
        for code_dir in CODE_DIRS:
            if not Path(code_dir.rstrip("/")).exists():
                continue

            code, stdout, stderr = self.run_command(["ruff", "check", code_dir])

            if code != 0:
                self.errors.append(f"Linting errors in {code_dir}:\n{stdout}\n{stderr}")
                print(f"FAIL {code_dir} has linting issues")
                print(f"Fix with: ruff check --fix {code_dir}")
                return False
            else:
                print(f"PASS {code_dir} linting OK")

        self.success_count += 1
        print("All linting passed (CI-strict)")
        return True

    def check_formatting(self) -> bool:
        """Check code formatting with ruff - exactly as strict as CI."""
        print("Checking code formatting (CI-strict)...")
        self.total_checks += 1

        # Check each directory individually to match CI behavior exactly
        for code_dir in CODE_DIRS:
            if not Path(code_dir.rstrip("/")).exists():
                continue

            code, stdout, stderr = self.run_command(
                ["ruff", "format", "--check", code_dir]
            )

            if code != 0:
                self.errors.append(
                    f"Formatting issues in {code_dir}:\n{stdout}\n{stderr}"
                )
                print(f"FAIL {code_dir} has formatting issues")
                print(f"Fix with: ruff format {code_dir}")
                return False
            else:
                print(f"PASS {code_dir} formatting OK")

        self.success_count += 1
        print("All formatting passed (CI-strict)")
        return True

    def check_type_checking(self) -> bool:
        """Check type annotations with mypy."""
        print("Checking type annotations...")
        self.total_checks += 1

        code, stdout, stderr = self.run_command(["mypy", "src/skdr_eval/"])

        if code != 0:
            self.errors.append(f"Type checking errors found:\n{stdout}\n{stderr}")
            return False

        self.success_count += 1
        print("Type checking passed")
        return True

    def check_tests(self) -> bool:
        """Run tests and check coverage."""
        print("Running tests with coverage...")
        self.total_checks += 1

        code, stdout, stderr = self.run_command(
            [
                "pytest",
                "-v",
                "--cov=skdr_eval",
                "--cov-report=json",
                "--cov-report=term-missing",
            ]
        )

        if code != 0:
            self.errors.append(f"Tests failed:\n{stdout}\n{stderr}")
            return False

        # Check coverage
        coverage_file = self.repo_root / "coverage.json"
        if coverage_file.exists():
            try:
                with coverage_file.open() as f:
                    coverage_data = json.load(f)
                    total_coverage = coverage_data.get("totals", {}).get(
                        "percent_covered", 0
                    )

                    if total_coverage < COVERAGE_THRESHOLD:
                        self.errors.append(
                            f"Coverage too low: {total_coverage:.1f}% (minimum: 80%)"
                        )
                        return False
                    else:
                        print(f"Tests passed with {total_coverage:.1f}% coverage")
            except (json.JSONDecodeError, KeyError) as e:
                self.warnings.append(f"Could not parse coverage report: {e}")
        else:
            self.warnings.append("Coverage report not found")

        self.success_count += 1
        return True

    def check_documentation(self) -> bool:
        """Check documentation requirements."""
        print("Checking documentation...")
        self.total_checks += 1

        # Check for docstrings in new/modified Python files using AST
        code, stdout, _ = self.run_command(
            ["git", "diff", "--name-only", "develop...HEAD"]
        )
        if code == 0:
            python_files = [
                f
                for f in stdout.strip().split("\n")
                if f.endswith(".py") and f.startswith("src/")
            ]

            for file_path in python_files:
                full_path = self.repo_root / file_path
                if full_path.exists():
                    try:
                        with full_path.open(encoding="utf-8") as f:
                            content = f.read()

                        # Use AST to check for missing docstrings
                        try:
                            tree = ast.parse(content, filename=str(full_path))
                        except SyntaxError:
                            self.warnings.append(
                                f"File {file_path} could not be parsed for docstring check (syntax error)"
                            )
                            continue

                        missing_docstrings = []

                        # Check module-level docstring
                        if ast.get_docstring(tree) is None:
                            missing_docstrings.append("module")

                        # Check functions and classes
                        for node in ast.walk(tree):
                            if (
                                isinstance(
                                    node,
                                    (
                                        ast.FunctionDef,
                                        ast.AsyncFunctionDef,
                                        ast.ClassDef,
                                    ),
                                )
                                and ast.get_docstring(node) is None
                                and not node.name.startswith("_")
                            ):
                                missing_docstrings.append(
                                    f"{type(node).__name__.lower()} '{node.name}'"
                                )

                        if missing_docstrings:
                            self.warnings.append(
                                f"File {file_path} is missing docstrings for: {', '.join(missing_docstrings)}"
                            )
                    except Exception as e:
                        self.warnings.append(
                            f"Error checking docstrings in {file_path}: {e}"
                        )

        self.success_count += 1
        print("Documentation check completed")
        return True

    def check_commit_messages(self) -> bool:
        """Check commit message format."""
        print("Checking commit messages...")
        self.total_checks += 1

        # Get commits ahead of develop
        code, stdout, _ = self.run_command(["git", "log", "--oneline", "develop..HEAD"])
        if code != 0:
            self.warnings.append("Could not check commit messages")
            return True

        commits = stdout.strip().split("\n") if stdout.strip() else []

        for commit in commits:
            if commit:
                message = commit.split(" ", 1)[1] if " " in commit else commit
                if not any(
                    message.startswith(prefix) for prefix in CONVENTIONAL_PREFIXES
                ):
                    self.warnings.append(
                        f"Commit message may not follow conventional format: '{message}'"
                    )

        self.success_count += 1
        print("Commit message check completed")
        return True

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("Starting contribution validation...\n")

        checks = [
            self.check_git_status,
            self.check_linting,
            self.check_formatting,
            self.check_type_checking,
            self.check_tests,
            self.check_documentation,
            self.check_commit_messages,
        ]

        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                self.errors.append(f"Error running {check.__name__}: {e}")
                all_passed = False
            print()  # Add spacing between checks

        return all_passed

    def print_summary(self):
        """Print validation summary."""
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        print(f"Passed: {self.success_count}/{self.total_checks} checks")

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
            print("\nFix these errors before submitting your PR!")
        else:
            # Get current branch name for better UX
            code, stdout, _ = self.run_command(["git", "branch", "--show-current"])
            current_branch = stdout.strip() if code == 0 else "<branch-name>"

            print("\nAll checks passed! Your contribution is ready for PR submission.")
            print("\nNext steps:")
            print(f"   1. Push your branch: git push origin {current_branch}")
            print("   2. Create PR targeting 'develop' branch")
            print("   3. Fill out the PR template completely")
            print("   4. Wait for CI checks and code review")


def main():
    """Main entry point."""
    validator = ContributionValidator()

    # Run validation
    success = validator.validate_all()
    validator.print_summary()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
