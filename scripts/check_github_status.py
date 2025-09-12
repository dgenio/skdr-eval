#!/usr/bin/env python3
"""
Script to check GitHub check runs status for the current commit.
This replaces the inline Python scripts used for checking CI status.
"""

import json
import subprocess
import sys
from typing import Dict, List, Any


def get_current_commit_sha() -> str:
    """Get the current commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit SHA: {e}")
        sys.exit(1)


def get_check_runs(owner: str, repo: str, commit_sha: str) -> Dict[str, Any]:
    """Get check runs for a specific commit from GitHub API."""
    import urllib.request
    
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}/check-runs"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        print(f"Error fetching check runs: {e}")
        sys.exit(1)


def analyze_check_runs(check_runs: Dict[str, Any]) -> None:
    """Analyze and display check runs status."""
    print(f"Total check runs: {check_runs['total_count']}")
    print()
    
    for run in check_runs['check_runs']:
        name = run['name']
        status = run['status']
        conclusion = run['conclusion']
        
        print(f"Check: {name}")
        print(f"  Status: {status}")
        print(f"  Conclusion: {conclusion}")
        
        if run.get('html_url'):
            print(f"  URL: {run['html_url']}")
        
        if conclusion == 'failure' and run.get('output', {}).get('text'):
            print(f"  Error details: {run['output']['text'][:200]}...")
        
        print()


def main():
    """Main function."""
    # Get repository info from git remote
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        remote_url = result.stdout.strip()
        # Extract owner/repo from URL like https://github.com/owner/repo.git
        if "github.com" in remote_url:
            parts = remote_url.split("/")
            owner = parts[-2]
            repo = parts[-1].replace(".git", "")
        else:
            print("Error: Not a GitHub repository")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error getting remote URL: {e}")
        sys.exit(1)
    
    # Get current commit SHA
    commit_sha = get_current_commit_sha()
    print(f"Checking status for commit: {commit_sha}")
    print(f"Repository: {owner}/{repo}")
    print()
    
    # Get and analyze check runs
    check_runs = get_check_runs(owner, repo, commit_sha)
    analyze_check_runs(check_runs)


if __name__ == "__main__":
    main()