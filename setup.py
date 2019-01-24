import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE", "r") as fh:
    license = fh.read()

from arbdmodel.version import get_version

setuptools.setup(
    name="arbdmodel",
    version=get_version(),
    author="Christopher Maffeo",
    author_email="cmaffeo2@illinois.edu",
    description="Python interface to ARBD simulation engine",
    license=license,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.engr.illinois.edu/tbgl/tools/arbdmodel",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=(
        'numpy>=1.14',
        'appdirs>=1.4'
    ),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: UIUC Open source License",
        "Operating System :: OS Independent",
    ),
)
