{
  description = "Log Vector development environments";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in {
      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.hello;
        });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
          pythonPkgs = pkgs.python312Packages;
          venvDir = ".venv";
          runtimeLibs = with pkgs; [
            stdenv.cc.cc.lib
            zlib
            openssl
            libffi
          ];
          runtimeLibPath = pkgs.lib.makeLibraryPath runtimeLibs;
        in {
          default = pkgs.mkShell {
            packages =
              [
                python
                pythonPkgs.pip
                pythonPkgs.virtualenv
                pkgs.cmake
                pkgs.pkg-config
                pkgs.gcc
                pkgs.gnumake
                pkgs.git
                pkgs.curl
                pkgs.openssl
                pkgs.libffi
              ]
              ++ pkgs.lib.optional (pkgs ? ollama) pkgs.ollama;

            shellHook = ''
              # Keep shell usable even if commands fail
              set -uo pipefail

              # Allow binary wheels (numpy, chromadb, torch) to find libstdc++ and CUDA libs on NixOS
              echo "Dev shell: setting LD_LIBRARY_PATH for GCC/CUDA runtime libs"
              echo "Dev shell: runtimeLibPath=${runtimeLibPath}"

              if [ -d /run/opengl-driver/lib ]; then
                CUDA_LIB_DIR=/run/opengl-driver/lib
                export LD_LIBRARY_PATH="$CUDA_LIB_DIR:${runtimeLibPath}:''${LD_LIBRARY_PATH:-}"
                export CUDA_HOME=/run/opengl-driver
              else
                export LD_LIBRARY_PATH="${runtimeLibPath}:''${LD_LIBRARY_PATH:-}"
              fi

              echo "Dev shell: LD_LIBRARY_PATH after export=$LD_LIBRARY_PATH"

              VENV_PATH="${"$"}{VENV_PATH:-${venvDir}}"

              if [ ! -d "$VENV_PATH" ]; then
                echo "Creating Python virtual environment at $VENV_PATH"
                python3 -m venv "$VENV_PATH"
              fi

              # shellcheck disable=SC1090
              source "$VENV_PATH/bin/activate"

              if [ -f requirements.txt ]; then
                req_hash="$(sha256sum requirements.txt | cut -d' ' -f1)"
                hash_file="$VENV_PATH/.requirements-hash"
                existing_hash=""
                if [ -f "$hash_file" ]; then
                  existing_hash="$(cat "$hash_file")"
                fi
                if [ "$existing_hash" != "$req_hash" ]; then
                  python -m pip install --upgrade pip setuptools wheel
                  python -m pip install -r requirements.txt
                  echo "$req_hash" > "$hash_file"
                fi
              fi

              export PYTHONPATH="$PWD:$PYTHONPATH"
            '';
          };
        });
    };
}
