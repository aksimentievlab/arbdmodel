#!/usr/bin/env python
"""
Setup script for arbdmodel package.
"""

from setuptools import setup, find_packages

# Try to get version from version.py
try:
    from arbdmodel.version import get_version
    version = get_version()
except ImportError:
    version = '0.1.0'

setup(
    name="arbdmodel",
    version=version,
    description="Advanced Rigid-Body Dynamics Modeling Package",
    author="ARBD Model Contributors",
    author_email="example@example.com",
    url="https://github.com/yourusername/arbdmodel",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "MDAnalysis>=2.0.0",
        "parmed>=3.4.3",
    ],
    extras_require={
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'gendocs>=0.4.0',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
