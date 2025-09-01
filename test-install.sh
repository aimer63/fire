#!/usr/bin/env bash

INSTALL_METHOD=$1
VERSION=$2

# Build the wheel locally
if [ "$INSTALL_METHOD" = "local" ]; then
  rm -f dist/firecast-*-py3-none-any.whl
  rm -rf build/ dist/ *.egg-info
  if ! python3 -m build --wheel; then
    echo "ERROR: Wheel build failed. Aborting." >&2
    exit 1
  fi
fi

# Remove the image if it exists
docker image rm -f firecast-test 2>/dev/null || true

# Build the image with the specified install method and version
docker build -t firecast-test \
  --build-arg INSTALL_METHOD="$INSTALL_METHOD" \
  --build-arg VERSION="$VERSION" .

# Run and remove the container
# docker run --rm -it firecast-test /bin/bash
docker run --rm -it firecast-test /bin/bash -c "python -c 'import firecast; print(firecast.__version__)'; exec /bin/bash"
