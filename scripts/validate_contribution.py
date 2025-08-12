#!/usr/bin/env python3
"""
Validation script for AI agents and developers.

This script performs comprehensive checks before submitting PRs to ensure
compliance with the development guidelines and prevent CI failures.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Coverage threshold constant
COVERAGE_THRESHOLD = 80


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
        if current_branch in ["main", "develop"]:
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
        """Check code linting with ruff."""
        print("Checking code linting...")
        self.total_checks += 1

        code, stdout, stderr = self.run_command(
            ["ruff", "check", "src/", "tests/", "examples/"]
        )

        if code != 0:
            self.errors.append(f"Linting errors found:\n{stdout}\n{stderr}")
            return False

        self.success_count += 1
        print("Linting passed")
        return True

    def check_formatting(self) -> bool:
        """Check code formatting with ruff."""
        print("Checking code formatting...")
        self.total_checks += 1

        code, stdout, stderr = self.run_command(
            ["ruff", "format", "--check", "src/", "tests/", "examples/"]
        )

        if code != 0:
            self.errors.append(f"Formatting issues found:\n{stdout}\n{stderr}")
            print("💡 Run 'make format' or 'ruff format src/ tests/ examples/' to fix")
            return False

        self.success_count += 1
        print("Formatting passed")
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

        # Check for docstrings in new/modified Python files
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
                    with full_path.open(encoding="utf-8") as f:
                        content = f.read()

                    # Simple check for docstrings (could be more sophisticated)
                    if (
                        "def " in content
                        and '"""' not in content
                        and "'''" not in content
                    ):
                        self.warnings.append(
                            f"File {file_path} may be missing docstrings"
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
        conventional_prefixes = [
            "feat:",
            "fix:",
            "docs:",
            "style:",
            "refactor:",
            "test:",
            "chore:",
        ]

        for commit in commits:
            if commit:
                message = commit.split(" ", 1)[1] if " " in commit else commit
                if not any(
                    message.startswith(prefix) for prefix in conventional_prefixes
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
            print("\nAll checks passed! Your contribution is ready for PR submission.")
            print("\nNext steps:")
            print("   1. Push your branch: git push origin <branch-name>")
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
