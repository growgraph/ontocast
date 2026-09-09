{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;
in
pkgs.mkShell {
  name = "ontocast-dev-shell";

  buildInputs = with pkgs; [
    python
    uv

    git
    cmake
    stdenv.cc.cc
    stdenv.cc.libcxx

    pkg-config
    graphviz
    cairo

    tesseract
    poppler
    opencv
    libGL
    mesa
    libglvnd
    glib

    # X11 runtime deps
    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXinerama
    xorg.libXi
    xorg.libxcb

    # BLAS/LAPACK for NumPy/hdbscan
    openblas
    lapack

    # Python build tools
    python.pkgs.setuptools
    python.pkgs.wheel
    python.pkgs.cython
    python.pkgs.numpy
    python.pkgs.scipy
  ];

  shellHook = ''
    # Ensure we're using Nix Python (3.12)
    export UV_PYTHON="${python}/bin/python"
    export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

    # Make libstdc++ visible to Python C extensions
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.xorg.libxcb.out}/lib:${pkgs.mesa}/lib:${pkgs.libglvnd}/lib:${pkgs.glib.out}/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LIBRARY_PATH"

    # Create venv if missing
    if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
      echo "Creating uv virtualenv $UV_PROJECT_ENVIRONMENT..."
      uv venv --python "$UV_PYTHON" "$UV_PROJECT_ENVIRONMENT"
    fi
    # A venv built on some other interpreter needs no handling here: UV_PYTHON
    # is exported above, so the sync below rebuilds the environment on the Nix
    # python by itself.

    # "Activate" without sourcing: just put the venv first in PATH
    export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
    export PATH="$VIRTUAL_ENV/bin:$PATH"

    # Sync project deps. `dev` is an extra in pyproject.toml, not a dependency
    # group, and --all-extras already covers it; `--group dev` aborts the sync
    # outright, which left every shell with an unsynced venv.
    echo "Syncing dependencies with uv..."
    if ! uv sync --all-extras; then
      # A failed sync scrolls past between the other banners, and the shell
      # still opens, so the next command fails somewhere unrelated.
      echo "warning: uv sync failed -- the venv is incomplete." >&2
    fi

    echo "🐍 ontocast uv dev environment ready"
    echo "Python: $(python --version)"
    echo "uv: $(uv --version)"
  '';
}
