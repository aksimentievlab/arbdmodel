# ARBD Model Documentation

This directory contains the documentation for the ARBD Model package.

## Documentation Structure

- `source/`: Contains the source files for the documentation
- `build/`: Will contain the built documentation (HTML, PDF, etc.)

## Building the Documentation

### Prerequisites

First, install the required dependencies:

```bash
pip install sphinx sphinx-rtd-theme gendocs
```

### Generating Documentation Files

The documentation is automatically generated from the package's docstrings and structure using `gendocs`. To regenerate the documentation source files:

```bash
python generate_docs.py
```

This will create/update the RST files in the `docs/source` directory.

### Building HTML Documentation

To build the HTML documentation:

```bash
cd docs
sphinx-build -b html source build/html
```

The HTML documentation will be available in the `docs/build/html` directory.

### Building PDF Documentation (Optional)

To build PDF documentation (requires LaTeX):

```bash
cd docs
sphinx-build -b latex source build/latex
cd build/latex
make
```

The PDF documentation will be available as `build/latex/ARBDModel.pdf`.

## ReadTheDocs Integration

This documentation is designed to work with ReadTheDocs. When you push to the repository, ReadTheDocs will automatically build the documentation if configured correctly.

### Setting up ReadTheDocs

1. Go to [ReadTheDocs](https://readthedocs.org/) and sign in or create an account
2. Import your repository
3. Configure the settings:
   - Set the documentation type to "Sphinx"
   - Set the Python version to "3.x"
   - Enable "Install project in a virtualenv with setup.py install"
   - Add "sphinx_rtd_theme" to the list of requirements

## Customizing the Documentation

- To modify the structure of the documentation, edit the `generate_docs.py` file
- To change the appearance or behavior of the Sphinx build, edit the `docs/source/conf.py` file
- To add custom pages, use the `gen.add_custom_page()` method in `generate_docs.py`
