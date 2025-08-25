FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip

ARG INSTALL_METHOD=release
ARG VERSION
ARG WHEEL_URL
ARG PYPI_PACKAGE=firecast

# Construct WHEEL_URL if not provided and VERSION is set
RUN if [ "$INSTALL_METHOD" = "release" ]; then \
  if [ -z "$WHEEL_URL" ] && [ -n "$VERSION" ]; then \
  WHEEL_URL="https://github.com/aimer63/fire/releases/download/v${VERSION}/firecast-${VERSION}-py3-none-any.whl"; \
  fi ; \
  pip install $WHEEL_URL ; \
  elif [ "$INSTALL_METHOD" = "pypi" ]; then \
  if [ -n "$VERSION" ]; then \
  pip install $PYPI_PACKAGE==$VERSION ; \
  else \
  pip install $PYPI_PACKAGE ; \
  fi ; \
  else \
  echo "Unknown INSTALL_METHOD: $INSTALL_METHOD" && exit 1 ; \
  fi

COPY configs/config.toml configs/config.toml

RUN echo "INSTALL_METHOD: $INSTALL_METHOD, VERSION: $VERSION"

CMD ["fire", "--config", "configs/config.toml"]
