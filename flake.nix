{
  description = "Minimal CUDA-enabled Python development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      nvidia-smi = pkgs.writeShellScriptBin "nvidia-smi" ''
        exec ${pkgs.stdenv.cc.bintools.dynamicLinker} /usr/lib/wsl/lib/nvidia-smi "$@"
      '';
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [ python312 uv cudaPackages.cudatoolkit nvidia-smi ];
        UV_PYTHON_DOWNLOADS = "never";
        NIX_LD = pkgs.stdenv.cc.bintools.dynamicLinker;
        shellHook = ''
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:/usr/lib/wsl/lib:${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
