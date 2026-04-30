# -*- coding: utf-8 -*-

import os
import platform
import shutil
from pathlib import Path
from .logger import logger, get_resource_path

# Default binary names by platform
DEFAULT_BINARIES = {
    "arbd": {
        "windows": "arbd.exe",
        "linux": "arbd",
        "darwin": "arbd",  # macOS uses same binary name as Linux
    },
    "namd": {
        "windows": "namd2.exe",
        "linux": "namd2",
        "darwin": "namd2",  # macOS uses same binary name as Linux
    },
    "vmd": {
        "windows": "vmd.exe",
        "linux": "vmd",
        "darwin": "vmd",  # macOS uses same binary name as Linux
    },
    "hydropro": {
        "windows": "hydropro10-msd.exe",
        "linux": "hydropro10-lnx.exe",
        "darwin": "hydropro10-lnx.exe",  # HydroPro doesn't have a macOS-specific version, use Linux version
    },
    "apbs": {
        "windows": "apbs.exe",
        "linux": "apbs",
        "darwin": "apbs",  # macOS uses same binary name as Linux
    },
    "gmsh": {
        "windows": "gmsh.exe",
        "linux": "gmsh",
        "darwin": "gmsh",  # macOS uses same binary name as Linux
    },
}

# Resource directories where binaries might be bundled with the package
RESOURCE_DIRS = {
    "hydropro": "hydropro10",
}

# Singleton for binary path management
class _BinaryManager:
    def __init__(self):
        # Initialize empty paths dictionary
        self._binary_paths = {}
        
        # Determine platform
        self._platform = platform.system().lower()
        if self._platform not in ("windows", "linux", "darwin"):
            logger.warning(f"Unsupported platform: {self._platform}, using linux defaults")
            self._platform = "linux"
        
        # For binary path purposes, treat macOS (darwin) similar to Linux for many tools
        # that don't have macOS-specific versions

    def set_binary_path(self, binary_name, path):
        """
        Set the path for a specific binary.
        
        Args:
            binary_name: The name of the binary (e.g., 'arbd', 'hydropro')
            path: The full path to the binary
        """
        if path is None:
            if binary_name in self._binary_paths:
                del self._binary_paths[binary_name]
            return
            
        path = Path(path)
        self._binary_paths[binary_name.lower()] = path
    
    def get_binary_path(self, binary_name):
        """
        Get the path for a specific binary, searching if not explicitly set.
        
        Args:
            binary_name: The name of the binary (e.g., 'arbd', 'hydropro')
            
        Returns:
            Path to the binary if found, None otherwise
        """
        binary_name = binary_name.lower()
        
        # 1. Return already configured path if available
        if binary_name in self._binary_paths:
            return self._binary_paths[binary_name]
        
        # 2. Search for the binary
        binary_path = self._find_binary(binary_name)
        if binary_path:
            self._binary_paths[binary_name] = binary_path
            return binary_path
            
        # 3. Not found
        return "Warning: Binary not found"
    
    def _find_binary(self, binary_name):
        """
        Search for a binary in standard locations.
        
        Search order:
        1. System PATH
        2. Package resource directories
        3. Current directory
        
        Args:
            binary_name: The name of the binary to find
            
        Returns:
            Path to the binary if found, None otherwise
        """
        # Get the default binary name for the current platform
        if binary_name in DEFAULT_BINARIES:
            binary_filename = DEFAULT_BINARIES[binary_name][self._platform]
        else:
            # If not in our defaults, use the name as is
            binary_filename = binary_name
        
        # 1. Check in system PATH
        system_path = shutil.which(binary_filename)
        if system_path:
            return Path(system_path)
        
        # 2. Check in package resource directories
        if binary_name in RESOURCE_DIRS:
            resource_dir = get_resource_path(RESOURCE_DIRS[binary_name])
            resource_path = resource_dir / binary_filename
            if resource_path.exists():
                # Make binary executable if needed (Unix only)
                if self._platform != "windows" and not os.access(resource_path, os.X_OK):
                    os.chmod(resource_path, 0o755)
                return resource_path
        
        # 3. Check in current directory
        local_path = Path.cwd() / binary_filename
        if local_path.exists():
            return local_path
        
        # Not found
        return None
    
    def get_all_paths(self):
        """
        Get all configured binary paths.
        
        Returns:
            Dictionary of binary name to path
        """
        return self._binary_paths.copy()
    
    def export_to_dict(self):
        """
        Export all binary paths as a dictionary of strings.
        
        Returns:
            Dictionary of binary name to string path
        """
        return {name: str(path) for name, path in self._binary_paths.items()}
    
    def import_from_dict(self, paths_dict):
        """
        Import binary paths from a dictionary.
        
        Args:
            paths_dict: Dictionary of binary name to path
        """
        for name, path in paths_dict.items():
            if path:  # Skip empty paths
                self.set_binary_path(name, path)
                
    def is_binary_essential(self, binary_name):
        """
        Check if a binary is considered essential for core functionality.
        
        Args:
            binary_name: The name of the binary to check
            
        Returns:
            True if the binary is essential, False otherwise
        """
        essential_binaries = ["arbd"]  # These are required for core functionality
        return binary_name.lower() in essential_binaries

# Create the singleton instance
BinaryManager = _BinaryManager()
"""
BinaryManager Example: 
    >>> arbd_path = BinaryManager.get_binary_path("arbd")
    >>> print(f"ARBD binary path: {arbd_path}")
    >>> BinaryManager.set_binary_path("hydropro", "/path/to/hydropro")
"""

# Initialize binary paths from environment variables
def initialize_binary_paths():
    """
    Initialize binary paths from environment variables.
    This function should be called during package initialization.
    If users don't wish to call this function, they can set the paths manually.
    """
    # Check environment variables for binary paths
    for binary in DEFAULT_BINARIES:
        env_var = f"{binary.upper()}_PATH"
        if env_var in os.environ:
            BinaryManager.set_binary_path(binary, os.environ[env_var])
    
    # Try to load from configuration file
    try:
        import json
        config_path = Path.home() / ".config" / "arbd" / "binaries.json"
        if config_path.exists():
            with open(config_path) as f:
                paths_dict = json.load(f)
                BinaryManager.import_from_dict(paths_dict)
    except Exception as e:
        # Silently continue with environment or defaults if config loading fails
        pass

# Initialize binary paths on module import
initialize_binary_paths()
