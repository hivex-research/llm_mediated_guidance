from pathlib import Path
from setuptools import setup, find_packages


reqs_dir = Path("./requirements")


def read_requirements(filename: str):
    requirements_file = reqs_dir / filename
    if requirements_file.is_file():
        return requirements_file.read_text().splitlines()
    else:
        return []


requirements_base = read_requirements("base.txt")

setup(
    name="human_intervention_marl",
    # version="1.0",
    license="Apache 2.0",
    license_files=["LICENSE"],
    url="[ANONYMIZED]",
    download_url="[ANONYMIZED]",
    author="[ANONYMIZED]",
    author_email="[ANONYMIZED]",
    description=(""),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8",
    install_requires=requirements_base,
)
