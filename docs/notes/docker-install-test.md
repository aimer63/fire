
# Testing Installation with Docker

This document describes how to use the `test-install.sh` script to test various installation methods for the `firecast` package using Docker.

## Overview

The `test-install.sh` script automates the process of building a Docker image (`firecast-test`) to verify that the `firecast` package can be installed correctly through different distribution channels. It uses the `Dockerfile` in the project root, which is designed to handle multiple installation strategies.

After a successful build, the script runs a container, prints the installed `firecast` version, and drops you into an interactive `bash` shell inside the container for further inspection.

## Usage

The script is executed from the project root directory.

```bash
./test-install.sh <INSTALL_METHOD> [VERSION]
```

### Arguments

- `INSTALL_METHOD`: (Required) Specifies the installation method to test.
- `VERSION`: (Optional) The specific package version to install. Its behavior depends on the chosen `INSTALL_METHOD`.

## Installation Methods

The script supports the following installation methods, which are passed as build arguments to the `Dockerfile`.

### 1. `local`

This method tests installation from a locally built Python wheel (`.whl`).

- The script first cleans any previous builds and creates a new wheel file in the `dist/` directory.
- The `Dockerfile` copies the wheel into the image and installs it using `pip`.
- The `VERSION` argument is ignored for this method.

**Example:**

```bash
./test-install.sh local
```

### 2. `pypi`

This method tests installation from the Python Package Index (PyPI).

- If a `VERSION` is provided, it attempts to install that specific version (e.g., `pip install firecast==<VERSION>`).
- If no `VERSION` is provided, it installs the latest version available on PyPI.

**Examples:**

```bash
# Install the latest version from PyPI
./test-install.sh pypi

# Install a specific version from PyPI
./test-install.sh pypi v0.1.0
```

### 3. `release`

This method tests installation from a GitHub release.

- If a `VERSION` is provided (e.g., `v0.1.0`), it constructs the download URL for the wheel from the corresponding GitHub release.
- If no `VERSION` is provided, the `Dockerfile` automatically fetches the tag name of the latest release from the GitHub repository and uses it to download the wheel.

**Examples:**

```bash
# Install from the latest GitHub release
./test-install.sh release

# Install from a specific GitHub release
./test-install.sh release v0.1.0
```
