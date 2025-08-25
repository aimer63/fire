#!/usr/bin/env bash

INSTALL_METHOD=$1
VERSION=$2

# Remove the image if it exists
docker image rm -f test-install 2>/dev/null || true

# Build the image with the specified install method and version
docker build -t test-install \
  --build-arg INSTALL_METHOD="$INSTALL_METHOD" \
  --build-arg VERSION="$VERSION" .

# Run and remove the container
docker run --rm -it test-install /bin/bash
