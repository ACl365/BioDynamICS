"""
Script to push changes to GitHub repository.

This script uses the GitHub integration module to push the changes
to the GitHub repository.
"""

import os
import sys
from src.github_integration import GitHubIntegration

def main():
    """Main function to push changes to GitHub."""
    # Get repository path
    repo_path = os.path.dirname(os.path.abspath(__file__))
    
    # Create GitHub integration
    github = GitHubIntegration(repo_path=repo_path)
    
    # Get current status
    print("Current repository status:")
    status = github.get_status()
    print(status)
    
    # Add all files
    print("\nAdding all files...")
    github.add_files()
    
    # Commit changes
    print("\nCommitting changes...")
    commit_message = "Add system integration module and related components"
    github.commit(commit_message)
    
    # Push changes
    print("\nPushing changes to GitHub...")
    github.push()
    
    print("\nChanges pushed to GitHub successfully!")

if __name__ == "__main__":
    main()