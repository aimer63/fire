FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip

ARG WHEEL_URL=https://github.com/aimer63/fire/releases/download/v0.1.1/firecast-0.1.1-py3-none-any.whl
ARG PYPI_PACKAGE=firecast

# Install from wheel or PyPI
RUN if [ "$WHEEL_URL" != "" ]; then \
  pip install $WHEEL_URL ; \
  else \
  pip install $PYPI_PACKAGE ; \
  fi

COPY configs/config.toml configs/config.toml

CMD ["fire", "--config", "configs/config.toml"]
