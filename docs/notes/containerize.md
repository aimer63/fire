Yes, running a `ProcessPoolExecutor` inside a container has significant implications. The good news is that it works perfectly well, but you must be aware of how container resource limits affect it. If you don't account for them, you can get either poor performance or unexpected crashes.

Here is a breakdown of the key implications and how to manage them.

### 1. CPU Core Allocation (The Most Important Implication)

This is the most critical point. Your code currently determines the number of worker processes like this:

```python
# firestarter/main.py
max_workers = multiprocessing.cpu_count()
# ...
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # ...
```

The behavior of `multiprocessing.cpu_count()` (which is the same as `os.cpu_count()`) inside a container can be misleading.

- **The Problem:** By default, a container might be _limited_ to a certain number of CPUs, but `os.cpu_count()` might still report the **total number of CPUs on the host machine**. For example, if you run a container on a 32-core server but limit it to 2 CPUs (`--cpus=2`), `os.cpu_count()` could still return `32`. Your application would then spawn 32 worker processes, all fighting for time on only 2 available cores. This leads to massive context switching and terrible performance, a phenomenon known as "oversubscription."

- **The Modern Solution:** Modern versions of Python (3.13+) and container runtimes are much smarter about this. They are generally "cgroup aware." This means `os.cpu_count()` is more likely to correctly report the number of CPUs allocated to the container, not the host. However, relying on this can still be brittle across different environments and Python versions.

- **The Robust Approach:** To be safe and explicit, you should not rely solely on `os.cpu_count()`. The best practice is to allow the number of workers to be configurable, for example, via an environment variable. This gives the person running the container full control.

  **Recommendation:**

  1.  Modify your code to read the number of workers from an environment variable, falling back to `cpu_count()` as a default.
  2.  When running the container, explicitly set this environment variable to match the CPU limit you assign to the container.

### 2. Shared Memory (`/dev/shm`)

The `multiprocessing` module in Python heavily uses shared memory for inter-process communication (IPC) to pass data between the main process and the worker processes. In Linux, this is handled by the `/dev/shm` filesystem.

- **The Problem:** By default, Docker allocates a very small size for `/dev/shm` (typically just **64MB**). If your simulation passes large objects (like the configuration or initial state) to the worker processes, you could exhaust this limited shared memory space, causing your simulation to hang or crash with cryptic errors.

- **The Solution:** When you run your container, you can and should increase the size of `/dev/shm`.

  - For Docker: `docker run --shm-size=1g ... your-image`
  - For Podman: `podman run --shm-size=1g ... your-image`

  A size of 1GB (`1g`) is a safe starting point for most applications.

### 3. Memory (RAM) Limits

When you create a `ProcessPoolExecutor`, you are creating new, separate Python processes. Each of these processes consumes its own memory.

- **The Problem:** The total memory consumed by your application will be roughly:
  `(Memory of main process) + (Number of workers * Memory of one worker process)`

  If you run your container with a memory limit (e.g., `docker run -m 2g ...`), and the total memory usage of all your Python processes exceeds this limit, the container's Out-Of-Memory (OOM) killer will abruptly terminate your application without a clean Python traceback.

- **The Solution:**
  1.  Be mindful of how much memory a single simulation run consumes.
  2.  Set a reasonable memory limit on your container that accounts for all the worker processes.
  3.  If memory usage is high, you may need to limit the number of workers to stay within the container's memory budget, even if you have many CPU cores available.

### Summary: How to Do It Right

Deploying your application in a container is a great idea for portability and reproducibility. To do it successfully with `ProcessPoolExecutor`:

1.  **Containerize Your App:** Write a `Dockerfile`. Use a slim base image (e.g., `python:3.11-slim`) and multi-stage builds to keep the image size small.
2.  **Control Your Workers:** Don't blindly trust `cpu_count()`. Make the number of workers configurable via an environment variable (e.g., `MAX_WORKERS`).
3.  **Run with Explicit Resource Limits:** When you execute your container, use the flags to control resources and match them with your application's configuration.

**Example `docker run` command:**

```bash
docker run \
  --rm -it \
  --cpus="4" \
  -m "2g" \
  --shm-size="1g" \
  -e MAX_WORKERS=4 \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/output:/app/output \
  your-simulation-image \
  configs/config.toml
```

By following these principles, you can ensure your multi-process Python application runs efficiently, reliably, and predictably inside a container.
