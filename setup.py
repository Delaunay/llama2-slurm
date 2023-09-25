#!/usr/bin/env python
import os
from pathlib import Path

from setuptools import setup


def get_requirements(path: str):
    return [req.strip() for req in open(path)]


with open("llama2/core/__init__.py") as file:
    for line in file.readlines():
        if "version" in line:
            version = line.split("=")[1].strip().replace('"', "")
            break

assert (
    os.path.exists(os.path.join("llama2", "__init__.py")) is False
), "llama2 is a namespace not a module"

extra_requires = {"plugins": ["importlib_resources"]}
extra_requires["all"] = sorted(set(sum(extra_requires.values(), [])))

if __name__ == "__main__":
    setup(
        name="llama2",
        version=version,
        extras_require=extra_requires,
        description="llama2 utilities and wrappers",
        long_description=(Path(__file__).parent / "README.rst").read_text(),
        author="Delaunay",
        author_email="pierre@delaunay.io",
        license="BSD 3-Clause License",
        url="https://llama2.readthedocs.io",
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: OS Independent",
        ],
        packages=[
            "llama2.core",
            "llama2.plugins.example",
        ],
        setup_requires=["setuptools"],
        install_requires=get_requirements("requirements.txt"),
        namespace_packages=[
            "llama2",
            "llama2.plugins",
        ],
        package_data={
            "llama2.data": [
                "llama2/data",
            ],
        },
    )
