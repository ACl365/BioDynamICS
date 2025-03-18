"""
BioDynamICS - GitHub Integration Module

This module provides functionality for integrating with GitHub,
allowing automatic commits, pushes, and other GitHub operations
after major development stages.

Author: Alexander Clarke
Date: March 18, 2025
"""

import os
import subprocess
import logging
from datetime import datetime

class GitHubIntegration:
    """
    Provides GitHub integration functionality for the BioDynamICS project.
    """
    
    def __init__(self, repo_path=None, remote_name="origin", branch="main"):
        """
        Initialize the GitHub integration.
        
        Parameters:
        -----------
        repo_path : str, optional
            Path to the Git repository (default: current directory)
        remote_name : str, optional
            Name of the remote repository (default: "origin")
        branch : str, optional
            Branch to work with (default: "main")
        """
        self.repo_path = repo_path or os.getcwd()
        self.remote_name = remote_name
        self.branch = branch
        
        # Set up logging
        self.logger = logging.getLogger("GitHubIntegration")
        
        # Verify git is installed
        self._verify_git_installed()
        
        # Verify repository exists
        self._verify_repository()
    
    def _verify_git_installed(self):
        """Verify that git is installed and available."""
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            self.logger.info("Git is installed and available")
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.error("Git is not installed or not available in PATH")
            raise RuntimeError("Git is not installed or not available in PATH")
    
    def _verify_repository(self):
        """Verify that the specified path is a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout.strip() != "true":
                raise RuntimeError("Not a git repository")
            self.logger.info(f"Verified git repository at {self.repo_path}")
        except subprocess.SubprocessError:
            self.logger.error(f"Not a git repository: {self.repo_path}")
            raise RuntimeError(f"Not a git repository: {self.repo_path}")
    
    def _run_git_command(self, command, check=True):
        """
        Run a git command and return the result.
        
        Parameters:
        -----------
        command : list
            Git command as a list of strings
        check : bool, optional
            Whether to check for command success (default: True)
            
        Returns:
        --------
        subprocess.CompletedProcess
            Result of the command
        """
        try:
            result = subprocess.run(
                ["git"] + command,
                cwd=self.repo_path,
                check=check,
                capture_output=True,
                text=True
            )
            return result
        except subprocess.SubprocessError as e:
            self.logger.error(f"Git command failed: {e}")
            if check:
                raise
            return None
    
    def get_status(self):
        """
        Get the status of the repository.
        
        Returns:
        --------
        str
            Git status output
        """
        result = self._run_git_command(["status"])
        return result.stdout
    
    def add_files(self, files=None):
        """
        Add files to the git index.
        
        Parameters:
        -----------
        files : list or str, optional
            Files to add (default: all files)
            
        Returns:
        --------
        bool
            Success status
        """
        if files is None:
            # Add all files
            command = ["add", "."]
        elif isinstance(files, str):
            # Add a single file
            command = ["add", files]
        else:
            # Add multiple files
            command = ["add"] + list(files)
        
        try:
            self._run_git_command(command)
            self.logger.info(f"Added files to git index: {files or 'all'}")
            return True
        except subprocess.SubprocessError:
            self.logger.error(f"Failed to add files to git index: {files or 'all'}")
            return False
    
    def commit(self, message):
        """
        Commit changes to the repository.
        
        Parameters:
        -----------
        message : str
            Commit message
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            self._run_git_command(["commit", "-m", message])
            self.logger.info(f"Committed changes: {message}")
            return True
        except subprocess.SubprocessError:
            self.logger.error(f"Failed to commit changes: {message}")
            return False
    
    def push(self):
        """
        Push changes to the remote repository.
        
        Returns:
        --------
        bool
            Success status
        """
        try:
            self._run_git_command(["push", self.remote_name, self.branch])
            self.logger.info(f"Pushed changes to {self.remote_name}/{self.branch}")
            return True
        except subprocess.SubprocessError:
            self.logger.error(f"Failed to push changes to {self.remote_name}/{self.branch}")
            return False
    
    def pull(self):
        """
        Pull changes from the remote repository.
        
        Returns:
        --------
        bool
            Success status
        """
        try:
            self._run_git_command(["pull", self.remote_name, self.branch])
            self.logger.info(f"Pulled changes from {self.remote_name}/{self.branch}")
            return True
        except subprocess.SubprocessError:
            self.logger.error(f"Failed to pull changes from {self.remote_name}/{self.branch}")
            return False
    
    def create_branch(self, branch_name):
        """
        Create a new branch.
        
        Parameters:
        -----------
        branch_name : str
            Name of the branch to create
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            self._run_git_command(["checkout", "-b", branch_name])
            self.branch = branch_name
            self.logger.info(f"Created and switched to branch: {branch_name}")
            return True
        except subprocess.SubprocessError:
            self.logger.error(f"Failed to create branch: {branch_name}")
            return False
    
    def switch_branch(self, branch_name):
        """
        Switch to an existing branch.
        
        Parameters:
        -----------
        branch_name : str
            Name of the branch to switch to
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            self._run_git_command(["checkout", branch_name])
            self.branch = branch_name
            self.logger.info(f"Switched to branch: {branch_name}")
            return True
        except subprocess.SubprocessError:
            self.logger.error(f"Failed to switch to branch: {branch_name}")
            return False
    
    def create_tag(self, tag_name, message=None):
        """
        Create a new tag.
        
        Parameters:
        -----------
        tag_name : str
            Name of the tag to create
        message : str, optional
            Tag message
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            if message:
                self._run_git_command(["tag", "-a", tag_name, "-m", message])
            else:
                self._run_git_command(["tag", tag_name])
            self.logger.info(f"Created tag: {tag_name}")
            return True
        except subprocess.SubprocessError:
            self.logger.error(f"Failed to create tag: {tag_name}")
            return False
    
    def push_tag(self, tag_name):
        """
        Push a tag to the remote repository.
        
        Parameters:
        -----------
        tag_name : str
            Name of the tag to push
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            self._run_git_command(["push", self.remote_name, tag_name])
            self.logger.info(f"Pushed tag {tag_name} to {self.remote_name}")
            return True
        except subprocess.SubprocessError:
            self.logger.error(f"Failed to push tag {tag_name} to {self.remote_name}")
            return False
    
    def commit_stage(self, stage_name, files=None, tag=True):
        """
        Commit changes for a development stage.
        
        This is a convenience method that adds files, commits with a
        standardized message, and optionally creates and pushes a tag.
        
        Parameters:
        -----------
        stage_name : str
            Name of the development stage
        files : list or str, optional
            Files to add (default: all files)
        tag : bool, optional
            Whether to create and push a tag (default: True)
            
        Returns:
        --------
        bool
            Success status
        """
        # Add files
        if not self.add_files(files):
            return False
        
        # Create commit message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Stage: {stage_name} - {timestamp}"
        
        # Commit changes
        if not self.commit(message):
            return False
        
        # Create and push tag if requested
        if tag:
            tag_name = f"stage-{stage_name.lower().replace(' ', '-')}"
            tag_message = f"Development stage: {stage_name}"
            
            if not self.create_tag(tag_name, tag_message):
                return False
            
            if not self.push_tag(tag_name):
                return False
        
        # Push changes
        if not self.push():
            return False
        
        self.logger.info(f"Successfully committed stage: {stage_name}")
        return True

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create GitHub integration
    github = GitHubIntegration()
    
    # Get repository status
    status = github.get_status()
    print(f"Repository status:\n{status}")
    
    # Commit a stage
    github.commit_stage("Example Stage", tag=False)