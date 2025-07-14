# firestarter Installation Guide

This guide explains how to install the `firestarter` package from a GitHub release on **Linux**,
**macOS**, and **Windows**.

---

## 1. Download the Release

1. Go to the [GitHub Releases page](https://github.com/aimer63/fire/releases).
2. Download the latest `.whl` file (e.g., `firestarter-0.1.0b1-py3-none-any.whl`) to your computer.

---

## 2. Install Python (if needed)

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

## 3. Install firestarter

Open a terminal (Linux/macOS) or Command Prompt (Windows), navigate to the folder where you
downloaded the `.whl` file, and run:

```sh
pip install firestarter-0.1.0b1-py3-none-any.whl
```

Verify the installation:

```sh
python -c "import firestarter; print(firestarter.__version__)"
```

---

## 4. Run firestarter

You can run firestarter directly (make sure to set the environment variables first) or
with a shell script, see [Usage](docs/usage.md).

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python -m firestarter.main config.toml
```

For Windows (PowerShell):

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python -m firestarter.main config.toml
```

See usage [Usage](docs/usage.md) for details.

## 5. Upgrading firestarter

To upgrade to a newer version (e.g., from `v0.1.0b1` to `v0.1.0b2`):

1. Download the new `.whl` file from the [GitHub Releases page](https://github.com/<your-username>/<your-repo>/releases).
2. (Optional but recommended) Uninstall the old version:

   ```sh
   pip uninstall firestarter
   ```

3. Install the new version:

   ```sh
   pip install firestarter-0.1.0b2-py3-none-any.whl
   ```

4. Verify the installation:

   ```sh
   python -c "import firestarter; print(firestarter.__version__)"
   ```

---
