from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tdaphist",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.3.5",
        "matplotlib>=3.10.6",
        "gudhi>=3.11.0",
        "ripser>=0.6.14"
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
