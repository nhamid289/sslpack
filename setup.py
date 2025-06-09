# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path
import pathlib

import pkg_resources

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# requirements
with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='wslearn',
    version='0.1.0',
    description='A weakly-supervised learning library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nhamid289/wslearn',
    author='Nazeef Hamid, Nan Ye, Jonathan Wilton',
    author_email='nazeef.h.289@gmail.com, nan.ye@uq.edu.au, jonathan.wilton@uq.edu.au',
    keywords='pytorch semi-supervised-learning weakly-supervised-learning',
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.11',
)
