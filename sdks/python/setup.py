"""Setup for Sentinella Python SDK"""

from setuptools import setup, find_packages

setup(
    name="sentinella-sdk",
    version="0.1.0",
    description="Python SDK for Sentinella AI Gateway",
    author="Sentinella Team",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.25.2",
    ],
    python_requires=">=3.11",
)

