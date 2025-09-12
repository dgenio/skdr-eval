#!/usr/bin/env python3
"""
Script to validate project configuration files.
This replaces the inline Python scripts used for configuration validation.
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple


def check_python_syntax(filepath: str) -> Tuple[bool, str]:
    """Check Python file syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_python_files(directory: str) -> List[Tuple[str, bool, str]]:
    """Validate all Python files in a directory."""
    results = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                is_valid, error = check_python_syntax(filepath)
                results.append((filepath, is_valid, error))
    
    return results


def validate_pyproject_toml(filepath: str) -> Tuple[bool, str]:
    """Validate pyproject.toml file."""
    try:
        import tomllib
        with open(filepath, 'rb') as f:
            data = tomllib.load(f)
        
        # Check required fields
        project = data.get('project', {})
        required_fields = ['name', 'description', 'readme', 'license', 'authors']
        
        missing_fields = [field for field in required_fields if field not in project]
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        # Check build system
        build_system = data.get('build-system', {})
        if not build_system.get('build-backend'):
            return False, "Missing build backend"
        
        return True, "Valid pyproject.toml"
        
    except ImportError:
        return False, "tomllib not available (Python < 3.11)"
    except Exception as e:
        return False, f"Error parsing TOML: {e}"


def validate_yaml_file(filepath: str) -> Tuple[bool, str]:
    """Validate YAML file."""
    try:
        import yaml
        with open(filepath, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True, "Valid YAML"
    except ImportError:
        return False, "PyYAML not available"
    except Exception as e:
        return False, f"Error parsing YAML: {e}"


def main():
    """Main validation function."""
    print("🔍 Validating project configuration...")
    print()
    
    # Validate Python files
    print("📁 Checking Python files...")
    python_results = validate_python_files('src')
    python_results.extend(validate_python_files('tests'))
    
    python_errors = [r for r in python_results if not r[1]]
    if python_errors:
        print("❌ Python syntax errors found:")
        for filepath, _, error in python_errors:
            print(f"  {filepath}: {error}")
    else:
        print(f"✅ All {len(python_results)} Python files are syntactically correct")
    
    print()
    
    # Validate pyproject.toml
    print("📋 Checking pyproject.toml...")
    is_valid, message = validate_pyproject_toml('pyproject.toml')
    if is_valid:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    
    print()
    
    # Validate YAML files
    print("📄 Checking YAML files...")
    yaml_files = ['.pre-commit-config.yaml']
    for yaml_file in yaml_files:
        if os.path.exists(yaml_file):
            is_valid, message = validate_yaml_file(yaml_file)
            if is_valid:
                print(f"✅ {yaml_file}: {message}")
            else:
                print(f"❌ {yaml_file}: {message}")
    
    print()
    
    # Summary
    total_errors = len(python_errors)
    if total_errors == 0:
        print("🎉 All validations passed!")
        sys.exit(0)
    else:
        print(f"❌ Found {total_errors} validation errors")
        sys.exit(1)


if __name__ == "__main__":
    main()