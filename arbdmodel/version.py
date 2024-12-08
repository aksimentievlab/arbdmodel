# -*- coding: utf-8 -*-
# Author: Douglas Creager <dcreager@dcreager.net>
# This file is placed into the public domain.

# Calculates the current version number.  If possible, this is the
# output of “git describe”, modified to conform to the versioning
# scheme that setuptools uses.  If “git describe” returns an error
# (most likely because we're in an unpacked copy of a release tarball,
# rather than in a git working copy), then we fall back on reading the
# contents of the RELEASE-VERSION file.
#
# To use this script, simply import it your setup.py file, and use the
# results of get_git_version() as your package version:
#
# from version import *
#
# setup(
#     version=get_git_version(),
#     .
#     .
#     .
# )
#
#
# This will automatically update the RELEASE-VERSION file, if
# necessary.  Note that the RELEASE-VERSION file should *not* be
# checked into git; please add it to your top-level .gitignore file.
#
# You'll probably want to distribute the RELEASE-VERSION file in your
# sdist tarballs; to do this, just create a MANIFEST.in file that
# contains the following line:
#
#   include RELEASE-VERSION

__all__ = ("get_version")

from pathlib import Path
import subprocess
from subprocess import Popen, PIPE

_version_file = Path(__file__).parent / "RELEASE-VERSION"


def check_git_repository():
    try:
        remotes = subprocess.check_output(['git', 'remote', '-v'], stderr=subprocess.STDOUT)
        return b'arbdmodel.git' in remotes
    except:
        return False   

def call_git_describe(abbrev):
    try:
        p = Popen(['git', 'describe', '--tags', '--abbrev=%d' % abbrev],
                  stdout=PIPE, stderr=PIPE)
        p.stderr.close()
        line = p.stdout.readlines()[0]
        return line.strip().decode('utf-8')

    except:
        return None


def is_dirty():
    try:
        p = Popen(["git", "diff-index", "--name-only", "HEAD"],
                  stdout=PIPE, stderr=PIPE)
        p.stderr.close()
        lines = p.stdout.readlines()
        return len(lines) > 0
    except:
        return False


def read_release_version():
    try:
        f = open(_version_file, "r")

        try:
            version = f.readlines()[0]
            return version.strip()

        finally:
            f.close()

    except:
        return None


def write_release_version(version):
    f = open(_version_file, "w")
    f.write("%s\n" % version)
    f.close()

def get_version(abbrev=7):
    # Read in the version that's currently in RELEASE-VERSION.

    
    release_version = read_release_version()

    if not check_git_repository():
        return release_version
        # raise Exception(__function__ +" called from outside a version controlled source repository")


    # First try to get the current version using “git describe”.

    split_version = call_git_describe(abbrev).split("-")
    version = split_version[0]
    if len(split_version) > 1 and int(split_version[-2]) > 0:
        version += ".dev{}".format(int(split_version[-2]))


    # If that doesn't work, fall back on the value that's in
    # RELEASE-VERSION.

    if version is None:
        version = release_version

    # If we still don't have anything, that's an error.

    if version is None:
        raise ValueError("Cannot find the version number!")

    # If the current version is different from what's in the
    # RELEASE-VERSION file, update the file to be current.

    if version != release_version:
        write_release_version(version)

    # Finally, return the current version.

    return version


if __name__ == "__main__":
    print( get_version() )

def read_version_file(filename):
    with open(filename) as fh:
        version = fh.readlines()[0].strip()
    return version

class Citation():
    def __init__(self, author=None, title=None, journal=None, volume=None, number=None, pages=None, year=None, doi=None):
        self.author = author
        self.title = title
        self.journal = journal
        self.volume = volume
        self.number = number
        self.pages = pages
        self.year = year
        self.doi = doi

    def display(self):
        msg = ''
        if self.author: msg = f'{self.author}. '
        if self.title: msg = f'{msg}{self.title}. '
        if self.journal: msg = f'{msg}{self.journal} '
        if self.volume:
            msg = f'{msg} {self.volume}{"," if any((self.number, self.pages, self.year)) else "."} '
            if self.number:
                msg = f'{msg}({self.number}){"" if any((self.pages,self.year)) else "."} '
            if self.pages: msg = f'{msg}{self.pages}{"" if self.year else "."} '
        if self.year: msg = f'{msg}({self.year}). '
        if self.doi:
            if self.doi[:4] == 'http':
                msg = f'{msg}{self.doi}'
            else:
                msg = f'{msg}https://doi.org/{self.doi}'
        return msg

try:
    version = read_version_file(_version_file)
except:
    try:
        version = read_version_file(_version_file)
    except:
        version = "Unknown"
