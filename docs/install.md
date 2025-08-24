# firecast Installation Guide

This guide explains how to install the `firecast` package by cloning the repository or from a GitHub release on **Linux**, **macOS**, and **Windows**.

---

## 1 Install Python (if needed)

- **Linux:**  
  Most distributions come with Python 3.10+ pre-installed. If not, install it using your package
  manager.

  ```sh
  sudo apt install python3 python3-pip  # Debian/Ubuntu
  sudo pacman -S python python-pip      # Arch
  ```

- **macOS:**  
  Install Python 3.10+ from [python.org](https://www.python.org/downloads/) or with Homebrew:

  ```sh
  brew install python
  ```

- **Windows:**  
  Download and install Python 3.10+ from [python.org](https://www.python.org/downloads/windows/).  
  **Important:** During installation, check "Add Python to PATH".

---

## 2. Install from Source

1. Clone the repository:

   ```sh
   git clone https://github.com/aimer63/fire.git
   cd fire
   ```

2. (Optional but recommended) Create and activate a virtual environment:

   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in editable/development mode:

   ```sh
   pip install -e .
   ```

   or with development dependencies:

   ```sh
   pip install -e .[dev]
   ```

4. Verify the installation:

   ```sh
   python -c "import firecast; print(firecast.__version__)"
   ```

---

## 3. Install from GitHub Release

1. Go to the [GitHub Releases page](https://github.com/aimer63/fire/releases).
2. Download the latest `.whl` file (e.g., `firecast-0.1.0b1-py3-none-any.whl`) to your computer.

---

### 3.1 Install firecast

Open a terminal (Linux/macOS) or Command Prompt (Windows), navigate to the folder where you
downloaded the `.whl` file, and run:

```sh
pip install firecast-0.1.0b1-py3-none-any.whl
```

Verify the installation:

```sh
python -c "import firecast; print(firecast.__version__)"
```

---

## 4. Run firecast

You can run firecast directly using the installed CLI command:

```sh
fire -f config.toml
```

All required environment variables for thread limiting are automatically set by the program on all platforms (Linux, macOS, Windows).  
No need to set them manually.

See [Usage](../docs/usage.md) for details.

## 5. Upgrading firecast

### If installed from source (git)

1. Pull the latest changes:

   ```sh
   git pull
   ```

2. Reinstall in editable mode:

   ```sh
   pip install -e .
   ```

3. Verify the installation:

   ```sh
   python -c "import firecast; print(firecast.__version__)"
   ```

---

### If installed from a GitHub release

1. Download the new `.whl` file from the [GitHub Releases page](https://github.com/aimer63/fire/releases).
2. (Optional but recommended) Uninstall the old version:

   ```sh
   pip uninstall firecast
   ```

3. Install the new version:

   ```sh
   pip install firecast-0.1.0b2-py3-none-any.whl
   ```

4. Verify the installation:

   ```sh
   python -c "import firecast; print(firecast.__version__)"
   ```

---
