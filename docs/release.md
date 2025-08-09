# firestarter Release Process

This guide describes the recommended steps to create and publish a new release of `firestarter`
using GitHub Releases.

---

## 1. Update Version Numbers

- Edit both files to set the new version (e.g., `0.1.0b2`):

  - `pyproject.toml`:

    ```toml
    version = "0.1.0b2"
    ```

  - `firestarter/version.py`:

    ```python
    __version__ = "0.1.0b2"
    ```

---

## 2. Finalize Code and Documentation

- Ensure all code is committed and working.
- Update `README.md`, example configs, and documentation as needed.
- Confirm `LICENSE` and `pyproject.toml` metadata are correct.

---

## 3. Commit and Tag the Release

```sh
git add .
git commit -m "Prepare release v0.1.0b2"
git tag v0.1.0b2
git push
git push origin v0.1.0b2
```

If you are using Github and have set a workflow to automatically build and publish releases, you can skip the next steps.

---

## 4. Build the Distribution

(Inside your environment)

```sh
pip install build twine
python -m build
```

- This creates `.whl` and `.tar.gz` files in the `dist/` directory.

---

## 5. (Optional) Test the Build Locally

```sh
pip install dist/firestarter-0.1.0b2-py3-none-any.whl
```

---

## 6. Create a GitHub Release

1. Go to your repository's **Releases** tab.
2. Click **"Draft a new release"**.
3. Set the tag (e.g., `v0.1.0b2`) and title (e.g., `firestarter v0.1.0b2`).
4. Write concise release notes (features, fixes, known issues).
5. Attach your `.whl` and `.tar.gz` files from the `dist/` directory.
6. Publish the release.

---

## 7. (Optional) Upload to PyPI or TestPyPI

If you want to distribute via PyPI:

```sh
twine upload --repository testpypi dist/*
# Or for real PyPI:
# twine upload dist/*
```

---

## 8. Announce and Share

- Share the GitHub release link with users and testers.
- Provide installation instructions (see `install.md`).

---

## Notes

- Do **not** commit or push the `dist/` directory to git; use GitHub Releases for distributing build
  artifacts.
- Always keep `pyproject.toml` and `version.py` in sync.
- For more details on installation, see [install.md](install.md).

---
