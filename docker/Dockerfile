FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
# Vivado/Vitis require en_US.UTF-8; a missing locale aborts HLS (exit 134).
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

ARG XRT_DEB_VERSION=xrt_202220.2.14.354_22.04-amd64-xrt
ARG POETRY_VERSION=2.1.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    g++ \
    git \
    git-lfs \
    libboost-all-dev \
    libncurses5 \
    libncurses5-dev \
    libtinfo5 \
    locales \
    make \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    unzip \
    wget \
    zip \
    zlib1g-dev \
    gosu \
    libudev1 \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && curl -fsSL https://curl.se/ca/cacert.pem -o /usr/local/share/ca-certificates/mozilla-cacert.crt \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="/opt/poetry/bin:${PATH}"
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION="${POETRY_VERSION}" python3.10 -

RUN wget -q -U 'Mozilla/5.0' \
    "https://www.xilinx.com/bin/public/openDownload?filename=${XRT_DEB_VERSION}.deb" \
    -O "/tmp/${XRT_DEB_VERSION}.deb" \
    && apt-get update \
    && apt-get install -y "/tmp/${XRT_DEB_VERSION}.deb" \
    && rm -f "/tmp/${XRT_DEB_VERSION}.deb" \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace /finn_build /finn_deps

COPY docker/entrypoint.sh /usr/local/bin/finn-entrypoint.sh
RUN chmod +x /usr/local/bin/finn-entrypoint.sh

WORKDIR /workspace

# Poetry creates /workspace/.venv (POETRY_VIRTUALENVS_IN_PROJECT=true).
ENV FINN_BUILD_DIR=/finn_build
ENV FINN_DEPS=/finn_deps
ENV FINN_XILINX_PATH=/tools/Xilinx
ENV NUM_DEFAULT_WORKERS=4
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

ENTRYPOINT ["/usr/local/bin/finn-entrypoint.sh"]
CMD ["bash"]
