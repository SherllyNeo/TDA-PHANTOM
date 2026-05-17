from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tdaphantom",
    version="1.0.1",
    description="Statistical hypothesis testing for persistence diagrams and barcodes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="W. Moriarty",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "matplotlib>=3.7",
        "gudhi>=3.11.0",
        "ripser>=0.6.14",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
