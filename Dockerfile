# hadolint global ignore=DL3013,DL3042,DL3008
FROM python:3.12-slim

WORKDIR /app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

ARG INSTALL_METHOD=release
ARG VERSION
ARG WHEEL_URL
ARG PYPI_PACKAGE=firecast

COPY dist/firecast-*.whl /app/

RUN if [ "$INSTALL_METHOD" = "release" ]; then \
      if [ -z "$VERSION" ]; then \
        VERSION=$(curl -s https://api.github.com/repos/aimer63/fire/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'); \
      fi; \
      if [ -z "$WHEEL_URL" ]; then \
        WHEEL_URL="https://github.com/aimer63/fire/releases/download/${VERSION}/firecast-${VERSION#v}-py3-none-any.whl"; \
      fi; \
      pip install "$WHEEL_URL"; \
    elif [ "$INSTALL_METHOD" = "local" ]; then \
      pip install /app/firecast-*-py3-none-any.whl; \
    elif [ "$INSTALL_METHOD" = "pypi" ]; then \
      if [ -n "$VERSION" ]; then \
        pip install "$PYPI_PACKAGE==$VERSION"; \
      else \
        pip install "$PYPI_PACKAGE"; \
      fi; \
    else \
      echo "Unknown INSTALL_METHOD: $INSTALL_METHOD" && exit 1; \
    fi

COPY configs/config.toml config.toml

RUN echo "INSTALL_METHOD: $INSTALL_METHOD, VERSION: $VERSION"
