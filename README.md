# speeling-good

A minimal Python and PyTorch project with CUDA support.

On x86-64 Linux with Nix, direnv, and a compatible NVIDIA driver installed:

```sh
git add flake.nix flake.lock
direnv allow
uv sync
nvidia-smi
uv run python main.py
```

Alternatively, enter the environment with `nix develop` instead of direnv. The
shell includes the CUDA toolkit and uses the host NVIDIA driver; on WSL it also
exposes `/usr/lib/wsl/lib`. PyTorch uses CUDA 12.8 wheels, which require a
compatible host driver. GPU access must be enabled in the host or WSL session.
The `nvidia-smi` wrapper expects the WSL driver utility at
`/usr/lib/wsl/lib/nvidia-smi`.

On NixOS, uv-installed executables such as Ruff and ty also require `nix-ld`.
Enable `programs.nix-ld.enable = true;` in the NixOS configuration and rebuild
the system, then reload the development shell. The shell sets `NIX_LD` and the
library path for these executables.
