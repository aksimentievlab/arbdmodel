#!/usr/bin/env python3
"""
Documentation generator for the arbdmodel package using sphinx-book-theme.

This script generates ReStructuredText (RST) documentation files
from the arbdmodel package's structure and configures Sphinx with
the sphinx-book-theme for a more modern look and feel.
"""

import os
import sys
import inspect
import importlib
import shutil
from pathlib import Path
import subprocess

# Define the package name
PACKAGE_NAME = "arbdmodel"

# Configure documentation paths
DOCS_DIR = Path("docs")
SOURCE_DIR = DOCS_DIR / "source"
BUILD_DIR = DOCS_DIR / "build"

# Define the structure of documentation sections and modules
DOCUMENTATION_STRUCTURE = {
    "Core": [
        "arbdmodel.core_objects",
        "arbdmodel.model",
        "arbdmodel.config",
        "arbdmodel.utils",
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
    "RigidBody Models": [
        "arbdmodel.rb_from_pdb",
        "arbdmodel.rb_from_mesh",
        "arbdmodel.rb_from_parametric",
        "arbdmodel.rb_contact_model",
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
    section_dir = output_dir / section_name.lower().replace(' ', '_')
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
    return section_dir.name

def create_main_index(sections, section_dirs, output_dir):
    """Create the main index.rst file."""
    content = [
        "ARBD Model Documentation",
        "=======================",
        "",
        "Welcome to the documentation for ARBD Model!",
        "",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "   :caption: Contents:",
        "",
    ]
    
    # Add each section to the toctree
    for section, dir_name in zip(sections, section_dirs):
        content.append(f"   {dir_name}/index")
        
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

def create_conf_py(source_dir):
    """Create the Sphinx configuration file with sphinx-book-theme."""
    conf_content = """# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'ARBD Model'
copyright = '2025, ARBD Model Team'
author = 'ARBD Model Team'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Theme options
html_theme_options = {
    "repository_url": "https://github.com/username/arbdmodel",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
    "home_page_in_toc": True,
    "navbar_footer_text": "ARBD Model Documentation",
    "extra_navbar": "<div>Version: 1.0.0</div>",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
"""
    
    conf_path = source_dir / "conf.py"
    with open(conf_path, 'w') as f:
        f.write(conf_content)
    
    print(f"Created Sphinx configuration file: {conf_path}")

def create_doc_directories():
    """Create necessary directories for documentation."""
    DOCS_DIR.mkdir(exist_ok=True)
    SOURCE_DIR.mkdir(exist_ok=True)
    
    # Create _static directory for custom CSS/JS
    static_dir = SOURCE_DIR / "_static"
    static_dir.mkdir(exist_ok=True)
    
    # Create _templates directory for custom templates
    templates_dir = SOURCE_DIR / "_templates"
    templates_dir.mkdir(exist_ok=True)

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'sphinx',
        'sphinx-book-theme',
        'sphinx-copybutton'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("The following required packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nAfter installing the required packages, run this script again.")
        sys.exit(1)

def build_docs():
    """Build the documentation using Sphinx."""
    try:
        print("\nBuilding documentation...")
        subprocess.run(['sphinx-build', '-b', 'html', str(SOURCE_DIR), str(BUILD_DIR)], check=True)
        print(f"\nDocumentation successfully built. You can view it at: {BUILD_DIR / 'index.html'}")
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}")
    except FileNotFoundError:
        print("Could not find sphinx-build. Make sure Sphinx is installed and in your PATH.")

def main():
    """Generate documentation for the arbdmodel package using sphinx-book-theme."""
    # Check required packages
    check_requirements()
    
    # Create doc directories
    create_doc_directories()
    
    # Process each section
    section_dirs = []
    for section_name, module_names in DOCUMENTATION_STRUCTURE.items():
        dir_name = create_section_index(section_name, module_names, SOURCE_DIR)
        section_dirs.append(dir_name)
    
    # Create main index
    create_main_index(DOCUMENTATION_STRUCTURE.keys(), section_dirs, SOURCE_DIR)
    
    # Create Sphinx configuration with sphinx-book-theme
    create_conf_py(SOURCE_DIR)
    
    print(f"\nDocumentation files generated in {SOURCE_DIR}")
    
    # Build the documentation
    response = input("\nWould you like to build the HTML documentation now? (y/n): ")
    if response.lower() in ('y', 'yes'):
        build_docs()
    else:
        print("\nTo build the documentation later, run:")
        print(f"sphinx-build -b html {SOURCE_DIR} {BUILD_DIR}")

if __name__ == "__main__":
    main()

