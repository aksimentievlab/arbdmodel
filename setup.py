"""``arbdmodel``: A simple Python package with automatic documentation
"""

import setuptools

__version__ = '0.0.0'

with open("README.rst", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="arbdmodel",
    version=__version__,
    author="",
    author_email="",
    description="A simple Python package with automatic documentation",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com//arbdmodel",
    packages=setuptools.find_packages(),
    install_requires=[
        # TODO: add dependencies
    ],
    classifiers=(
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        'Natural Language :: English',
    ),
)
