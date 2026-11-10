# vvm — vLLM Version Manager

Manage multiple [vLLM](https://github.com/vllm-project/vllm) installations with isolated Python virtual environments. Switch between releases, commits, branches, and PRs instantly.

```bash
vvm install 0.26.0     # each version gets its own venv under ~/.vvm/versions/
vvm use 0.26.0         # switch the active version in your current shell
```

## Highlights

- **Isolated** — every version gets its own venv; no cross-version dependency conflicts.
- **Instant switching** — `vvm use` flips a symlink and updates `PATH`/`VIRTUAL_ENV` in place.
- **Any ref** — releases, `nightly`, `commit:`, `branch:`, `pr:`, a git URL, or a local checkout.
- **Precompiled by default** — pulls matching wheels from wheels.vllm.ai; compiles from source only when it has to.
- **Kernel-aware** — `--path` installs match the precompiled `_C.so` to your fork's merge-base, and repeat `--compile` builds reuse kernels when nothing that feeds them changed.
- **No runtime deps** — single static binary; needs only `uv` and `python3`.

## Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Environment Variables](#environment-variables)
- [Requirements](#requirements)
- [Contributing](#contributing)

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vvm/main/install.sh | bash
```

Downloads the prebuilt binary for your platform (Linux x86_64/aarch64) into `~/.local/bin` and adds shell integration (tab completion + in-place version switching) to your profile. Start a new terminal or `source ~/.bashrc` to activate. Options pass through `bash -s`, e.g. `... | bash -s -- --to ~/bin`.

### Manual install

If you'd rather not pipe to bash, grab the binary and wire up the shell yourself:

```bash
curl -fsSL https://github.com/vllm-project/vvm/releases/latest/download/vvm-$(uname -m)-linux-musl -o vvm && chmod +x vvm && mv vvm ~/.local/bin/
echo 'eval "$(vvm init bash)"' >> ~/.bashrc   # or: vvm init zsh / vvm init fish
```

### With Cargo

Builds from source; requires [Rust](https://rustup.rs/) stable.

```bash
cargo install --git https://github.com/vllm-project/vvm
```

### Build from source

```bash
git clone https://github.com/vllm-project/vvm.git
cd vvm
./install.sh            # builds release, installs to ~/.local/bin, configures your shell
# or: ./install.sh --to ~/bin
```

## Quick Start

```bash
vvm install 0.26.0            # Install a release version
vvm use 0.26.0                # Activate it
vllm -v                       # 0.26.0+cu130

vvm install nightly           # Install another version
vvm use nightly               # Switch to it
vvm ls                        # List installed versions
```

## Usage

### Install by version, commit, branch, or PR

```bash
vvm install 0.18.0                    # Release version
vvm install nightly                   # Latest nightly build
vvm install commit:abc123def          # Specific commit (uses wheels.vllm.ai)
vvm install branch:main               # Branch HEAD (falls back to recent commits if no wheel yet)
vvm install pr:1234                   # PR HEAD commit
vvm install 0.18.0 --python python3.12  # Pick the interpreter for the venv
```

Branch installs automatically fall back to the most recent commit on that branch with a precompiled wheel available on wheels.vllm.ai if the HEAD commit doesn't have one yet.

**Auto-update:** Re-running `vvm install branch:main`, `pr:1234`, or `nightly` for an already-installed version detects if a newer upstream commit exists and reinstalls in place. No need to `uninstall` first.

**Compile fallback:** When no usable precompiled wheel exists for the requested ref + arch + CUDA variant — either it's missing entirely, or it's stale relative to C++/CUDA changes in the requested branch/PR/commit — vvm prompts to clone and build from source. Pass `-y` / `--yes` to auto-accept (useful in scripts and CI). Prompt is skipped in non-TTY runs.

For `commit:X`, if the exact wheel is missing vvm first falls back to source mode (clones the commit + uses a nearby wheel for C++/CUDA) before offering compile. If that has drift and you decline to compile, the install errors out rather than silently using a nearby wheel — `commit:X` is a precise request.

### Install from a git URL

```bash
vvm install --repo https://github.com/org/vllm-fork            # Clone + install from a fork
vvm install --repo git@github.com:org/vllm-fork --branch mybranch
vvm install myfork --repo https://github.com/org/vllm-fork --branch mybranch --compile
```

`--repo` clones into `~/.vvm/repos/` (cached, reused on reinstall). Private repos work if you've already run `gh auth login` — vvm uses gh as a git credential helper so it doesn't re-prompt for a username. Combine with `--branch` (alias `--ref`; takes a branch, tag, or commit SHA — pin a SHA for reproducible installs) to select what to check out, or `--compile` to build from source.

### Install from a local repo

```bash
vvm install --path ~/vllm                          # Editable install with precompiled C++/CUDA
vvm install --path ~/vllm --compile                # Compile C++/CUDA from source
vvm install --path ~/vllm --branch myfeature       # Checkout a ref first (branch/tag/SHA), then install
vvm install my-kernel-fix --path ~/vllm --compile  # Custom name (default: local-{branch}-{hash})
vvm install --path ~/vllm --wheel-commit nightly   # Use nightly wheel for C++ libs
vvm install --path ~/vllm --wheel-commit abc123    # Use a specific commit's wheel
vvm install --path ~/vllm --wheel-location /tmp/vllm.whl  # Use a specific wheel file
vvm install --path ~/vllm --force                  # Reinstall (clears the existing venv first)
```

**Default (`--path` without `--compile`):** Editable install (`pip install -e`) using `VLLM_USE_PRECOMPILED=1`. The precompiled wheel commit is auto-matched to the age of your local repo's HEAD, so you get compatible C++ libraries. Python code changes take effect immediately without rebuilding.

**`--compile` mode:** Editable install that compiles the C++/CUDA extensions from source. Use this when you need to modify CUDA kernels; `.py` edits still take effect live, but changing kernels means re-running `--compile`. Automatically detects your GPU architecture via `nvidia-smi` and sets `TORCH_CUDA_ARCH_LIST` to skip irrelevant architectures. Build log is saved to `~/.vvm/versions/<name>/build.log`. You can override build settings via environment variables (`MAX_JOBS`, `NVCC_THREADS`, `TORCH_CUDA_ARCH_LIST`).

**Kernel reuse:** Repeat `--compile` installs of the same repo skip compilation entirely when it would produce identical binaries — i.e. no changes to `csrc/`, `cmake/`, or `CMakeLists.txt` (committed or uncommitted) since the last compile, same torch pin, and the previous build's arch list covers the local GPU. The kernels a previous `--compile` left in the source tree are reused, so a Python-only commit installs in seconds. Set `VVM_FORCE_COMPILE=1` to force a full rebuild. If reused kernels fail the post-install import check, vvm automatically recompiles from source.

**`--source` mode:** `vvm install <specifier> --source` clones the resolved commit and installs it with `VLLM_USE_PRECOMPILED=1`, borrowing C++/CUDA from the nearest wheel. This is what the compile fallback uses internally; pass it explicitly to force source mode for a ref that does have a wheel.

### CUDA variant selection

```bash
vvm install 0.18.0 --cuda cu130      # Specific CUDA variant
vvm install 0.18.0 --cuda cu128      # CUDA 12.8
vvm install 0.18.0 --cuda cpu        # CPU-only
# Auto-detected from nvidia-smi, /usr/local/cuda/version.json, or nvcc
# Works on login nodes where nvidia-smi is unavailable
```

When none of those report a CUDA version, vvm installs the **cpu** build and says so — a GPU wheel picked on no evidence would only fail later at import. On a host with no CUDA visible that will run somewhere with a GPU (a toolkit-less login node, a build container), pass `--cuda cu130` or set `$VVM_CUDA_VARIANT`.

### Manage versions

```bash
vvm ls                                # List installed versions (version string, branch name)
vvm ls --json                         # Machine-readable listing
vvm ls-remote                         # List available releases
vvm ls-remote --pre                   # Include pre-releases
vvm ls-remote 0.17                    # Filter by pattern
vvm current                           # Show active version
vvm info                              # Detailed info for current version (commit SHA, etc.)
vvm info branch:main                  # Detailed info for a specific version
vvm info --json                       # Machine-readable info
vvm uninstall 0.17.1                  # Remove a version
vvm uninstall 0.17.1 commit-abc123 branch-main   # Remove multiple at once
```

### Aliases

```bash
vvm alias                             # List all aliases
vvm alias stable 0.18.0               # Create alias
vvm alias dev local-api-patch-abc123  # Alias a local build
vvm use stable                        # Switch via alias
vvm unalias stable                    # Remove alias
```

### Run commands in a specific version

```bash
vvm exec v0.18.0 -- vllm serve Qwen/Qwen3-0.6B
vvm exec pr:1234 -- python -c "import vllm; print(vllm.__version__)"
vvm exec branch:main -- pip install flash-attn   # pip → uv pip in the right venv
```

### Install extras into the active venv

Packages that can't install cleanly via plain `pip install` — needing git submodules, vendored CUDA sources, or matching companion wheels — have bundled recipes:

```bash
vvm install-extra deepgemm                        # DeepGEMM @ the active vllm's pin
vvm install-extra flashinfer                      # FlashInfer @ the active vllm's pin
vvm install-extra deepep                          # DeepEP + NVSHMEM @ the active vllm's pin

vvm install-extra deepgemm --ref main             # Latest main
vvm install-extra flashinfer --ref 0.6.7          # A release version
vvm install-extra flashinfer --ref v0.6.7         # ...or its tag, same thing
vvm install-extra flashinfer --ref nightly        # Prebuilt nightly wheels (flashinfer + cubin + jit-cache)
vvm install-extra deepep --ref d4f41e4e93         # A specific commit
vvm install-extra flashinfer --repo https://github.com/me/flashinfer-fork  # Your own fork
```

`--ref` takes a commit SHA, tag, branch, or a bare release version. Upstream tags releases `v0.6.7` while everyone says "0.6.7", so vvm tries the literal ref first — a fork that tags without the prefix keeps working — and only falls back to the `v`-prefixed form, telling you when it does. SHAs are never rewritten. A ref that exists nowhere in the repo fails immediately, with nearby tags listed, rather than after a clone:

```
$ vvm install-extra flashinfer --ref v0.67
error: could not resolve 'v0.67' as a branch, tag, or commit in https://github.com/flashinfer-ai/flashinfer.git.
       Close tags: v0.6.17rc1, v0.6.16rc5, v0.6.16rc4, v0.6.16rc3, v0.6.16rc2
```

**Version defaults follow the active vllm**, so an extra matches what that vllm was actually tested against rather than whatever upstream `main` happens to be today. `deepgemm` reads `DEEPGEMM_GIT_REF` from `tools/install_deepgemm.sh`; `deepep` and `flashinfer` read `DEEPEP_COMMIT_HASH` / `FLASHINFER_VERSION` from `docker/versions.json` — the manifest vllm's own release images build from. For `--path`/`--compile` installs the files are read straight from your source tree (works offline); for wheel installs they're fetched from GitHub at that install's commit or tag. Each install prints where its pin came from:

```
Installing DeepEP (https://github.com/deepseek-ai/DeepEP.git @ d4f41e4e93) [pin: active vllm]
                                                                                 ^ active vllm | user | bundled fallback
```

`--ref` always wins, and a vendored default covers the cases where neither source is readable (offline, or a vllm predating the pin file).

**`flashinfer`** installs prebuilt whenever it can. A release ref — including the default pin from vllm — comes straight from PyPI as a `py3-none-any` wheel, so it lands in seconds with no compile; `--ref nightly` pulls prebuilt nightly wheels from `flashinfer.ai/whl/nightly/`. Two companions are installed afterward, pinned to the same version (a mismatch makes flashinfer refuse to import). **`flashinfer-cubin`** is a vllm runtime requirement — `requirements/cuda.txt` pins it next to `flashinfer-python`, and since 0.6.14 it lives only on `flashinfer.ai/whl/`, not PyPI. **`flashinfer-jit-cache`** (~1.4 GB, CUDA- and arch-specific) is what vllm's Dockerfile adds on top; it buys startup latency, not correctness, so a miss is only a warning and `VVM_FLASHINFER_NO_JIT_CACHE=1` skips it — kernels then JIT on first use. Only a branch, a raw commit, or `--repo <fork>` actually needs a source build, which takes 10-20 minutes and is cached under `~/.vvm/cache/wheels/` keyed by commit + CUDA + arch list. Set `VVM_FLASHINFER_BUILD=1` to force the source build for a release ref.

Every flashinfer install path ends with a verification step: `flashinfer show-config` runs inside the venv, exercising the import, torch, the CUDA runtime, and the cubin/jit-cache wiring, and the identity lines are surfaced so a mismatch is visible immediately instead of mid-inference:

```
  Verify: FlashInfer version: 0.6.17rc1
  Verify: flashinfer-cubin version: 0.6.17rc1
  Verify: flashinfer-jit-cache: 0.6.17rc1
  Verify: Torch version: 2.11.0+cu130
  Verify: CUDA runtime available: Yes
✓ FlashInfer installed
```

A verification failure is a warning, not an error — show-config also talks to NVIDIA's cubin repository, and a network hiccup shouldn't fail wheels that are already installed. Run with `-v` to stream the full dump. Skipped on nodes without GPU access.

**`deepep`** downloads NVSHMEM from NVIDIA's CDN into `~/.vvm/cache/nvshmem/` (reused across installs), auto-detects your GPU arch for `TORCH_CUDA_ARCH_LIST`, and installs DeepEP against it. Multi-node IBGDA still needs a one-time `sudo` driver config step (vvm prints instructions after install).

Output is hidden behind a spinner unless `-v` / `VVM_VERBOSE=1` is set. Errors surface the tail of the build log.

### Install packages into the active venv

```bash
vvm pip install fastsafetensors                   # Runs uv pip inside the current version's venv
vvm pip uninstall flash-attn
```

### Per-project version pinning

Create a `.vvmrc` file in your project root:

```
0.18.0
```

Then `vvm use` with no arguments reads it. Supports all specifier types:

```
pr:1234
commit:abc123def
branch:main
```

## How It Works

Each installed version gets its own Python virtual environment under `~/.vvm/versions/`:

```
~/.vvm/
├── versions/
│   ├── v0.18.0/
│   │   ├── venv/            # Isolated Python venv with vllm + torch + deps
│   │   └── metadata.json    # Install type, CUDA variant, version string, timestamp
│   ├── local-main-abc123/   # Local repo install
│   ├── commit-abc123de/
│   ├── branch-main/
│   ├── pr-1234/
│   └── nightly/
├── aliases/                  # Symlinks (default, stable, etc.)
├── repos/                    # Cached git clones (used by --repo)
├── cache/                    # Persistent downloads (NVSHMEM tarball, built wheels)
└── current -> versions/v0.18.0
```

**Install strategies:**

| Specifier | Source | Strategy |
|-----------|--------|---------|
| `0.18.0` | wheels.vllm.ai + PyPI | `uv pip install vllm==0.18.0 --extra-index-url .../cu130` |
| `nightly` | wheels.vllm.ai/nightly | Direct wheel URL from metadata.json, fallback to recent main commits |
| `commit:abc123` | wheels.vllm.ai/{commit} | Direct wheel URL; on miss falls back to source mode (cloned commit + nearby wheel) |
| `branch:main` | GitHub API → commit → wheels.vllm.ai | Direct wheel URL; fallback walks back 30 recent branch commits |
| `pr:1234` | GitHub API → commit → wheels.vllm.ai | Direct wheel URL |
| `--path ~/vllm` | Local repo | `VLLM_USE_PRECOMPILED=1 uv pip install -e ~/vllm` |
| `--path ~/vllm --compile` | Local repo | `uv pip install -e ~/vllm --no-build-isolation-package vllm` (kernels built from source) |

**Version switching** updates the `current` symlink. The shell hook (`vvm init bash`) wraps the `vvm` binary so that `vvm use` automatically updates `PATH` and `VIRTUAL_ENV` in your current shell. Fish integration uses `vvm env --shell fish` to generate its activation commands; `vvm env` defaults to Bash/Zsh syntax.

**Variant verification:** After install, vvm checks that the installed vllm wheel matches the requested CUDA variant. Skipped on login nodes where GPU driver is unavailable.

**Shared filesystems:** Version directories are safe to keep on NFS/Lustre. Concurrent installs from multiple nodes coordinate through a `hostname:pid` lock with an mtime heartbeat, the `current` symlink is swapped atomically, and directory removal works around NFS `.nfs*` silly-rename files.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VVM_DIR` | Override data directory (default: `~/.vvm`) |
| `VVM_CUDA_VARIANT` | Override CUDA variant auto-detection |
| `VVM_VERBOSE` | Set to any value for verbose pip/git/build output (disables spinner) |
| `VVM_FORCE_COMPILE` | Force a full kernel rebuild on `--compile` instead of reusing cached kernels |
| `VVM_NO_WHEEL_CACHE` | Bypass the `install-extra` built-wheel cache |
| `VVM_NVSHMEM_VER` | Override NVSHMEM version for `install-extra deepep` (default: `3.3.24`) |
| `GITHUB_TOKEN` / `GH_TOKEN` | GitHub API auth (avoids rate limits; for private repos use `gh auth login` instead) |

**Flags for `--path` installs** (mapped to vllm build env vars):

| Flag | Env Var | Description |
|------|---------|-------------|
| `--cuda` | `VLLM_PRECOMPILED_WHEEL_VARIANT` | CUDA variant for precompiled wheel |
| `--wheel-commit` | `VLLM_PRECOMPILED_WHEEL_COMMIT` | Override commit for precompiled wheel. Can be `nightly` |
| `--wheel-location` | `VLLM_PRECOMPILED_WHEEL_LOCATION` | Exact wheel URL or local path |

## Requirements

- Linux (vllm only supports Linux natively)
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (for package installation)
- [Rust toolchain](https://rustup.rs/) (only to build vvm from source)
- NVIDIA GPU + CUDA driver (or use `--cuda cpu`)

## Contributing

Development setup and architecture notes live in [AGENTS.md](AGENTS.md).
Shell integration tests require `fish` and `zsh` in addition to Bash.

```bash
git config core.hooksPath .githooks   # reject unformatted commits locally
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) and must be signed off under the [Developer Certificate of Origin](DCO):

```bash
git commit -s -m "fix: ..."
```

## License

[Apache-2.0](LICENSE)
