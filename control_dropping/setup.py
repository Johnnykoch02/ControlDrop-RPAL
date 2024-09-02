""" Setup
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()


def _read_reqs(relpath):
    fullpath = path.join(path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [
            s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))
        ]


REQUIREMENTS = _read_reqs("requirements.txt")
exec(open(path.join(here, "src/control_dropping_rpal/version.py")).read())
setup(
    name="control_dropping_rpal",
    version=__version__,
    description="Control Dropping RPAL",
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    url="https://github.com/johnnykoch02/ControlDropping",
    author="",
    author_email="",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Note that this is a string of words separated by whitespace, not a list.
    keywords="Control Dropping robotics",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    python_requires=">=3.8",
)
