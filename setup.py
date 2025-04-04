"""
Setup script for CleanSlate.
"""

import os
import re
from setuptools import setup, find_packages

# Get the version from cleanslate/__init__.py
with open(os.path.join("cleanslate", "__init__.py"), "r") as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string in cleanslate/__init__.py")

# Get the long description from the README file
with open("README.md", "r") as f:
    long_description = f.read()

# Core dependencies
install_requires = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "pyyaml>=6.0",
    "pydantic>=1.9.0",
    "joblib>=1.1.0",
    "typing-extensions>=4.0.0",
]

# Optional dependencies
extras_require = {
    # For anomaly detection
    "anomaly": [
        "tensorflow>=2.8.0",
        "pyod>=1.0.0",
        "alibi-detect>=0.10.0",
    ],
    # For data validation
    "validation": [
        "great-expectations>=0.15.0",
        "pandera>=0.13.0",
        "evidently>=0.2.0",
    ],
    # For API and backend
    "api": [
        "fastapi>=0.75.0",
        "uvicorn>=0.17.0",
        "python-multipart>=0.0.5",
        "sqlalchemy>=1.4.0",
        "alembic>=1.7.0",
    ],
    # For UI
    "ui": [
        "streamlit>=1.10.0",
        "plotly>=5.6.0",
        "dash>=2.3.0",
    ],
    # For testing
    "test": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "hypothesis>=6.0.0",
    ],
    # For documentation
    "docs": [
        "sphinx>=4.4.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
    # For development
    "dev": [
        "black>=22.1.0",
        "isort>=5.10.0",
        "flake8>=4.0.0",
        "mypy>=0.931",
    ],
    # Full installation (everything)
    "full": [],
}

# Build full dependencies
extras_require["full"] = sorted(list(set(
    dep for deps in extras_require.values() for dep in deps
)))

setup(
    name="cleanslate",
    version=version,
    description="Enterprise-grade data cleaning solution for ML pipelines and applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CleanSlate Team",
    author_email="info@cleanslate.ai",
    url="https://github.com/cleanslateteam/cleanslate",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "cleanslate=main:main",
        ],
    },
    include_package_data=True,
) 