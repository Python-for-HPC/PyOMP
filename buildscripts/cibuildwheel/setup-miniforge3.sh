#!/usr/bin/env bash

set -euxo pipefail

MINIFORGE_VERSION="26.3.2-3"

# Read LLVM_VERSION from environment and error if not set
if [ -z "${LLVM_VERSION:-}" ]; then
    echo "Error: LLVM_VERSION environment variable is not set." >&2
    exit 1
fi

if [ "$(uname)" = "Darwin" ]; then
    OS_NAME="MacOSX"
else
    OS_NAME="Linux"
fi
ARCH_NAME="$(uname -m)"

case "${OS_NAME}-${ARCH_NAME}" in
    Linux-x86_64)
        MINIFORGE_SHA256="848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c"
        ;;
    Linux-aarch64)
        MINIFORGE_SHA256="2c113a69297e612b01ca0f320c22a3107a11f2ab9b573d79ac868a175945ce29"
        ;;
    Linux-ppc64le)
        MINIFORGE_SHA256="df7e80ee070ccc6031e2710eb4cf81ee0012264b587306b5aa3890d3d89edd97"
        ;;
    MacOSX-x86_64)
        MINIFORGE_SHA256="39273e4c89a0a1af4538010615d44ae8f44e1af41007e02def593d20f316b003"
        ;;
    MacOSX-arm64)
        MINIFORGE_SHA256="59168f1e24d0a4ad9932021170809fca836cd240e183eeeb331d5bcfc0098168"
        ;;
    *)
        echo "Unsupported Miniforge platform: ${OS_NAME}-${ARCH_NAME}" >&2
        exit 1
        ;;
esac

echo "Installing Miniforge..."
MINIFORGE_INSTALLER="Miniforge3-${MINIFORGE_VERSION}-${OS_NAME}-${ARCH_NAME}.sh"
MINIFORGE_RELEASE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}"
MINIFORGE_URL="${MINIFORGE_RELEASE_URL}/${MINIFORGE_INSTALLER}"
mkdir -p _downloads
curl -fsSL "${MINIFORGE_URL}" -o _downloads/miniforge3.sh
if command -v sha256sum >/dev/null 2>&1; then
    printf '%s  %s\n' "${MINIFORGE_SHA256}" _downloads/miniforge3.sh | sha256sum --check -
elif command -v shasum >/dev/null 2>&1; then
    printf '%s  %s\n' "${MINIFORGE_SHA256}" _downloads/miniforge3.sh | shasum -a 256 --check -
else
    echo "No SHA256 verification command is available" >&2
    exit 1
fi
mkdir -p _stage
bash _downloads/miniforge3.sh -b -f -p "_stage/miniforge3"
echo "Miniforge installed"
source "_stage/miniforge3/bin/activate" base

# Create conda environment with tools and libraries for the LLVM_VERSION.
echo "Installing llvmdev ${LLVM_VERSION}..."
conda create -n llvmdev-${LLVM_VERSION} --override-channels -c conda-forge -q -y clang=${LLVM_VERSION} clangxx=${LLVM_VERSION} clang-tools=${LLVM_VERSION} llvmdev=${LLVM_VERSION} zstd
