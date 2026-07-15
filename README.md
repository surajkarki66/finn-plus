# FINN+

**FINN+** is a fork of [FINN](https://github.com/Xilinx/finn) — a dataflow compiler for fast, scalable quantized neural network inference on AMD/Xilinx FPGAs. It targets QNNs and generates customized dataflow architectures for high throughput and low latency on FPGAs.

- [Wiki documentation](https://github.com/eki-project/finn-plus/wiki)
- [EKI project](https://www.eki-project.tech/)

This repository includes a **Docker-based development environment** so you can work on FINN+ without installing Python dependencies and system packages directly on your host. Xilinx tools (Vivado/Vitis) stay on the host and are mounted into the container.

## Why Docker?


| On the host                 | Inside the container                     |
| --------------------------- | ---------------------------------------- |
| Ubuntu 24.04 (or similar)   | Ubuntu 22.04 (`finn-plus:22.04`)         |
| Vivado / Vitis installation | Same path, read-only mount               |
| Docker + Docker Compose     | Poetry, Python 3.10, FINN+ dev deps, XRT |


The container handles Poetry install, `finn deps update`, and `finn check` on first start (see [wiki Quick Start — Option B](https://github.com/eki-project/finn-plus/wiki)).

## Prerequisites

**On the host:**

1. **Docker** with the **Compose plugin** (`docker compose version`)
2. **Xilinx tools** — Vivado and Vitis (tested with **2024.2**; 2022.2 is also supported upstream)
3. Enough disk space for build artifacts and dependencies (`finn_build` and `finn_deps` volumes)

**Not required on the host:** Python 3.10, Poetry, or FINN+ Python packages — the container sets those up.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/surajkarki66/finn-plus.git
cd finn-plus
```

This fork ([surajkarki66/finn-plus](https://github.com/surajkarki66/finn-plus)) is based on [eki-project/finn-plus](https://github.com/eki-project/finn-plus) and adds the Docker development workflow described below.

### 2. Configure environment

Copy the example env file and point it at your Xilinx installation:

```bash
cp docker/.env.example .env
```

Edit `.env` and set at least:

```bash
FINN_XILINX_PATH=/tools/Xilinx      # same absolute path as on your host
FINN_XILINX_VERSION=2024.2          # must match your installed version
```

`run-docker.sh` auto-detects your user/group IDs (`HOST_UID` / `HOST_GID`) so files created in the container are owned by your host user.

### 3. Start the development container

```bash
./run-docker.sh
```

This script will:

- Verify Docker and Xilinx paths
- Build the image `finn-plus:22.04` if needed
- Start a container with the repo mounted at `/workspace`
- Run first-time setup: `poetry install`, `finn deps update`, `finn check`
- Drop you into an interactive shell

First launch can take a while (Poetry install + dependency downloads).

### 4. Use FINN+ inside the container

```bash
# Verify setup
finn check

# Run tests (example)
finn test tests/util/test_config.py

# Run a YAML build
finn build path/to/build_config.yaml path/to/model.onnx

# Open a Jupyter notebook (notebooks extra is installed in dev mode)
jupyter notebook --ip=0.0.0.0 notebooks/
```

Project files under `/workspace` are the same as on your host — edits persist immediately.

**Key files:**


| File                           | Purpose                                                      |
| ------------------------------ | ------------------------------------------------------------ |
| `run-docker.sh`                | Host entry point — loads `.env`, checks Xilinx, runs Compose |
| `docker-compose.yml`           | Service definition, volumes, environment                     |
| `docker/Dockerfile`            | Ubuntu 22.04 image with Poetry, XRT, build tools             |
| `docker/entrypoint.sh`         | User mapping, Xilinx `settings64.sh`, dev setup              |
| `docker/.env.example`          | Template for `.env`                                          |
| `docker/settings.yaml.example` | Template copied to `settings.yaml` on first run              |


## Configuration

### Environment variables (`.env`)


| Variable                | Default         | Description                                                           |
| ----------------------- | --------------- | --------------------------------------------------------------------- |
| `FINN_XILINX_PATH`      | `/tools/Xilinx` | Host path to Xilinx tools (mounted at the same path in the container) |
| `FINN_XILINX_VERSION`   | `2024.2`        | Vivado/Vitis version folder name                                      |
| `NUM_DEFAULT_WORKERS`   | `4`             | Parallel workers for HLS / Vivado                                     |
| `HOST_UID` / `HOST_GID` | auto            | Match container file ownership to your host user                      |


### FINN settings (`settings.yaml`)

On first start, `docker/settings.yaml.example` is copied to `settings.yaml` in the repo root (gitignored). Adjust build dirs and workers there if needed.

### Force reinstall / re-check

Set these in `.env` to re-run setup steps:

```bash
FINN_FORCE_POETRY_INSTALL=1   # poetry install
FINN_FORCE_DEPS_UPDATE=1      # finn deps update
FINN_FORCE_CHECK=1            # finn check
```

## Common commands

**From the host** (any command after `./run-docker.sh` is passed into the container):

```bash
./run-docker.sh                    # interactive shell
./run-docker.sh finn check         # one-off command
./run-docker.sh finn test -k mobilenet
./run-docker.sh bash -c "cd tutorials/fpga_flow && python build.py"
```

**Inside the container:**

```bash
finn --help
finn deps update
finn build <config.yaml> <model.onnx>
poetry install                     # after pyproject.toml changes
```

**Rebuild the image** after Dockerfile changes:

```bash
docker compose build finn
```

## Troubleshooting

### Xilinx tools not found

```
ERROR: Xilinx tools not found at /tools/Xilinx
```

Install Vivado/Vitis on the host and set `FINN_XILINX_PATH` in `.env` to the real install path. The path **must be identical** inside and outside the container — Xilinx scripts use hardcoded absolute paths.

### Vivado version mismatch

If you see a warning about `settings64.sh`, check that `FINN_XILINX_VERSION` matches the folder under `$FINN_XILINX_PATH/Vivado/`.

### PYNQ board files download fails

If `finn deps update` cannot fetch PYNQ board files inside the container, run on the **host**:

```bash
./docker/fetch-pynq-boardfiles.sh
```

Board files are placed under `finn_deps/board_files/` in the repo.

### Permission issues on created files

Ensure `HOST_UID` and `HOST_GID` in `.env` match your host user (`id -u` / `id -g`). `run-docker.sh` sets these automatically if omitted.

### Clean build / dependency caches

```bash
docker compose down -v    # removes finn_build and finn_deps volumes
rm -rf .venv              # forces poetry reinstall on next start
```

## Project layout (high level)

```
finn-plus/
├── run-docker.sh          # Docker entry point
├── docker-compose.yml
├── docker/                # Dockerfile, entrypoint, env templates
├── src/finn/              # FINN+ Python package
├── finn-rtllib/           # RTL building blocks
├── models/                # Example models (DVC)
├── notebooks/             # Jupyter tutorials
├── tests/                 # Test suite
└── tutorials/             # Example flows (e.g. FPGA integration)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). For larger changes, check the [feature tracker](https://github.com/orgs/eki-project/projects/1) and open an issue first.

## License

BSD — see [LICENSE.txt](LICENSE.txt).

## About

FINN+ is maintained by researchers from [Paderborn University](https://en.cs.uni-paderborn.de/ceg) (CEG) and [PC²](https://pc2.uni-paderborn.de/) as part of the [eki research project](https://www.eki-project.tech/).