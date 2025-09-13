#!/usr/bin/env python3
"""
Script to create a GitHub release for v0.3.4.
This script provides the exact API call needed to create the release.
"""

import json
import subprocess
import sys
from typing import Dict, Any


def create_github_release() -> None:
    """Create a GitHub release using curl and GitHub API."""
    
    # Release data
    release_data = {
        "tag_name": "v0.3.4",
        "target_commitish": "main",
        "name": "v0.3.4: Fix pip publishing and code quality issues",
        "body": """This release includes critical fixes for pip publishing and significant code quality improvements:

## 🐛 Bug Fixes
- Fix pip publishing versioning issues with setuptools-scm
- Remove _version.py from version control to prevent conflicts
- Fix all 19 ruff linting errors (unused variables and regex patterns)

## 🔧 Improvements  
- Enhanced release workflow with better error handling
- Improved setuptools-scm configuration with local_scheme
- Added maintainable script files for project validation
- Better pre-commit configuration

## 📦 Technical Changes
- _version.py is now generated dynamically (not committed)
- All Python files pass linting checks
- Enhanced CI/CD workflows with better debugging
- Improved code maintainability and documentation

This release ensures that pip publishing works correctly and maintains high code quality standards.""",
        "draft": False,
        "prerelease": False
    }
    
    # Convert to JSON
    json_data = json.dumps(release_data, indent=2)
    
    print("🚀 Creating GitHub release for v0.3.4...")
    print()
    print("To create this release, you need to:")
    print("1. Go to: https://github.com/dgenio/skdr-eval/releases")
    print("2. Click 'Create a new release'")
    print("3. Fill in the following information:")
    print()
    print("Tag name: v0.3.4")
    print("Release title: v0.3.4: Fix pip publishing and code quality issues")
    print()
    print("Release description:")
    print(release_data["body"])
    print()
    print("4. Click 'Publish release'")
    print()
    print("Alternatively, if you have a GitHub token, you can run:")
    print()
    print("curl -X POST 'https://api.github.com/repos/dgenio/skdr-eval/releases' \\")
    print("  -H 'Accept: application/vnd.github.v3+json' \\")
    print("  -H 'Authorization: token YOUR_GITHUB_TOKEN' \\")
    print("  -d '" + json_data.replace('\n', '\\n') + "'")
    print()
    print("This will trigger the release workflow and publish to PyPI!")


def main():
    """Main function."""
    create_github_release()


if __name__ == "__main__":
    main()