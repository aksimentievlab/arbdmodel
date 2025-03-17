#!/usr/bin/env python3
"""
Documentation generator for the arbdmodel package.

This script manually generates ReStructuredText (RST) documentation files
from the arbdmodel package's structure. This is a custom implementation since
the gendocs API differs from the expected structure.
"""

import os
import sys
import inspect
import importlib
from pathlib import Path

# Define the package name
PACKAGE_NAME = "arbdmodel"

# Configure documentation paths
DOCS_DIR = Path("docs")
SOURCE_DIR = DOCS_DIR / "source"

# Define the structure of documentation sections and modules
DOCUMENTATION_STRUCTURE = {
    "Core": [
        "arbdmodel.core_objects",
        "arbdmodel.model",
        "arbdmodel.sim_config",
    ],
    "Polymer Modeling": [
        "arbdmodel.polymer",
        "arbdmodel.fjc_polymer_model",
        "arbdmodel.hps_polymer_model",
        "arbdmodel.kh_polymer_model",
        "arbdmodel.mpipi_polymer",
        "arbdmodel.onck_polymer_model",
        "arbdmodel.sali_polymer_model",
        "arbdmodel.ssdna_two_bead",
    ],
    "Structure Models": [
        "arbdmodel.structure_from_pdb",
        "arbdmodel.structure_rigidbody",
        "arbdmodel.mesh_process_volume",
        "arbdmodel.mesh_process_surface",
        "arbdmodel.mesh_rigidbody",
        "arbdmodel.simplearbd",
    ],
    "Interaction Potentials": [
        "arbdmodel.interactions",
        "arbdmodel.ibi",
    ],
    "Simulation Engines": [
        "arbdmodel.engine",
        "arbdmodel.parmed_bd",
    ],
    "Shape-Based Models": [
        "arbdmodel.shape_cg",
    ],
    "Utilities": [
        "arbdmodel.coords",
        "arbdmodel.grid",
        "arbdmodel.logger",
        "arbdmodel.version",
        "arbdmodel.binary_manager",
    ]
}

def create_module_rst(module_name, output_dir):
    """Create RST documentation for a Python module."""
    try:
        # Try to import the module
        module = importlib.import_module(module_name)
        
        # Extract module name for the title
        simple_name = module_name.split('.')[-1]
        
        # Start building the RST content
        content = [
            f"{simple_name} module",
            "=" * len(f"{simple_name} module"),
            "",
            f".. py:module:: {module_name}",
            "",
        ]
        
        # Add module docstring if available
        if module.__doc__:
            content.append(module.__doc__.strip())
            content.append("")
        
        # Add automodule directive
        content.extend([
            ".. automodule:: " + module_name,
            "   :members:",
            "   :undoc-members:",
            "   :show-inheritance:",
            ""
        ])
        
        # Write the RST file
        output_path = output_dir / f"{simple_name}.rst"
        with open(output_path, 'w') as f:
            f.write('\n'.join(content))
            
        print(f"Created module documentation: {output_path}")
        return simple_name
        
    except ImportError as e:
        print(f"Warning: Could not import module {module_name}: {e}")
        return None
    except Exception as e:
        print(f"Error processing module {module_name}: {e}")
        return None

def create_section_index(section_name, module_names, output_dir):
    """Create an index RST file for a section."""
    # Create section directory if it doesn't exist
    section_dir = output_dir / section_name
    section_dir.mkdir(exist_ok=True)
    
    # Create index content
    content = [
        f"{section_name}",
        "=" * len(section_name),
        "",
        ".. toctree::",
        "   :maxdepth: 4",
        "",
    ]
    
    # Process each module in the section
    for module_name in module_names:
        simple_name = create_module_rst(module_name, section_dir)
        if simple_name:
            content.append(f"   {simple_name}")
    
    # Write the index file
    index_path = section_dir / "index.rst"
    with open(index_path, 'w') as f:
        f.write('\n'.join(content))
        
    print(f"Created section index: {index_path}")

def create_main_index(sections, output_dir):
    """Create the main index.rst file."""
    content = [
        "ARBD Model Documentation",
        "========================",
        "",
        "Welcome to the documentation for ARBD Model!",
        "",
        "ARBD Model is an advanced rigid-body dynamics modeling and simulation package for biomolecular systems.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "   :caption: Contents:",
        "",
    ]
    
    # Add each section to the toctree
    for section in sections:
        content.append(f"   {section}/index")
        
    # Add indices and tables
    content.extend([
        "",
        "Indices and tables",
        "==================",
        "",
        "* :ref:`genindex`",
        "* :ref:`modindex`",
        "* :ref:`search`"
    ])
    
    # Write the index file
    index_path = output_dir / "index.rst"
    with open(index_path, 'w') as f:
        f.write('\n'.join(content))
        
    print(f"Created main index: {index_path}")

def main():
    """Generate documentation for the arbdmodel package."""
    # Ensure docs directories exist
    DOCS_DIR.mkdir(exist_ok=True)
    SOURCE_DIR.mkdir(exist_ok=True)
    
    # Process each section
    for section_name, module_names in DOCUMENTATION_STRUCTURE.items():
        create_section_index(section_name, module_names, SOURCE_DIR)
    
    # Create main index
    create_main_index(DOCUMENTATION_STRUCTURE.keys(), SOURCE_DIR)
    
    print(f"\nDocumentation generation complete. Documentation files are in {SOURCE_DIR}")
    
    # Suggest next steps
    print("\nNext steps:")
    print(" 1. Make sure conf.py exists in the docs/source directory")
    print(" 2. Run 'sphinx-build -b html docs/source docs/build/html' to build HTML docs")
    print(" 3. Set up ReadTheDocs to automatically build from your repository")

if __name__ == "__main__":
    main()
