# Installation Guide

This guide covers how to install CleanSlate and its dependencies.

## Requirements

CleanSlate requires:

- Python 3.8 or later
- pip (Python package installer)

## Basic Installation

You can install CleanSlate using pip:

```bash
pip install cleanslate
```

This will install the core CleanSlate package with basic functionality.

## Optional Dependencies

CleanSlate has several optional dependencies for additional features:

- `anomaly`: Dependencies for advanced anomaly detection
- `validation`: Dependencies for data validation
- `api`: Dependencies for API and backend
- `ui`: Dependencies for the web UI
- `test`: Dependencies for testing
- `docs`: Dependencies for documentation
- `dev`: Dependencies for development
- `full`: All optional dependencies

To install CleanSlate with optional dependencies, use:

```bash
# For web UI
pip install cleanslate[ui]

# For anomaly detection
pip install cleanslate[anomaly]

# For all features
pip install cleanslate[full]

# For multiple feature sets
pip install cleanslate[ui,anomaly]
```

## Installing from Source

To install CleanSlate from source:

```bash
# Clone the repository
git clone https://github.com/cleanslateteam/cleanslate.git
cd cleanslate

# Install in development mode
pip install -e .

# Install with all dependencies
pip install -e .[full]
```

## Verifying Installation

To verify that CleanSlate is installed correctly:

```bash
python -c "import cleanslate; print(cleanslate.__version__)"
```

You should see the current version of CleanSlate printed.

You can also check the CLI:

```bash
cleanslate version
```

## Troubleshooting

### Missing Dependencies

If you encounter errors about missing dependencies, try installing the specific feature set you need:

```bash
pip install cleanslate[ui]  # For UI-related errors
```

### Installation Fails

If installation fails, try:

1. Updating pip: `pip install --upgrade pip`
2. Installing in a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install cleanslate
   ```

### Platform-Specific Issues

#### Windows

On Windows, you might need to install additional build tools for some dependencies:

```bash
pip install --upgrade setuptools wheel
```

#### macOS

On macOS, you might need to install Xcode command-line tools:

```bash
xcode-select --install
```

## Next Steps

Now that you've installed CleanSlate, check out the [Quick Start Guide](quickstart.md) to begin using it. 