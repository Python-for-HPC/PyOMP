from __future__ import annotations

import os
from pathlib import Path
import subprocess

import modal


PYTHON_VERSIONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
NUMBA_VERSION = "0.63.1"
MINIFORGE_VERSION = "26.3.2-3"
MINIFORGE_SHA256 = "848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WHEEL_DIRECTORY = REPOSITORY_ROOT / "dist"
MINIFORGE_PREFIX = "/opt/miniforge3"


def find_linux_wheels() -> dict[str, Path]:
    if not WHEEL_DIRECTORY.is_dir():
        raise RuntimeError(f"Linux wheel directory does not exist: {WHEEL_DIRECTORY}")

    wheels = {}
    for python_version in PYTHON_VERSIONS:
        abi = python_version.replace(".", "")
        matches = sorted(WHEEL_DIRECTORY.glob(f"*cp{abi}-*x86_64.whl"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one cp{abi} Linux x86-64 wheel in {WHEEL_DIRECTORY}, "
                f"found {len(matches)}: {matches}"
            )
        wheels[python_version] = matches[0]
    return wheels


LINUX_WHEELS = find_linux_wheels()


def image_build_commands() -> tuple[list[str], list[str]]:
    environment_commands = []
    wheel_commands = []
    for python_version, wheel in LINUX_WHEELS.items():
        environment = f"py{python_version.replace('.', '')}"
        python = f"{MINIFORGE_PREFIX}/envs/{environment}/bin/python"
        environment_commands.extend(
            [
                (
                    f"{MINIFORGE_PREFIX}/bin/conda create -q -y "
                    f"--override-channels -c conda-forge -n {environment} "
                    f"python={python_version} pip"
                ),
                (
                    f"{python} -m pip install "
                    f"numba=={NUMBA_VERSION} lark cffi setuptools"
                ),
                (
                    f'{python} -c "import numba; '
                    f"assert numba.__version__ == '{NUMBA_VERSION}'\""
                ),
            ]
        )
        wheel_commands.append(
            f"{python} -m pip install --no-deps /wheels/{wheel.name}"
        )
    return environment_commands, wheel_commands


ENVIRONMENT_COMMANDS, WHEEL_COMMANDS = image_build_commands()


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("ca-certificates", "curl")
    .run_commands(
        f"curl -fsSL https://github.com/conda-forge/miniforge/releases/download/"
        f"{MINIFORGE_VERSION}/Miniforge3-{MINIFORGE_VERSION}-Linux-x86_64.sh "
        "-o /tmp/miniforge3.sh",
        f"printf '%s  %s\\n' {MINIFORGE_SHA256} /tmp/miniforge3.sh "
        "| sha256sum --check -",
        f"bash /tmp/miniforge3.sh -b -p {MINIFORGE_PREFIX}",
    )
    .run_commands(*ENVIRONMENT_COMMANDS)
    .add_local_dir(WHEEL_DIRECTORY, "/wheels", copy=True)
    .run_commands(*WHEEL_COMMANDS)
)

app = modal.App("pyomp-gpu-ci")


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60,
    restrict_modal_access=True,
    single_use_containers=True,
    block_network=True,
)
def test_gpu_wheels() -> None:
    subprocess.run(["nvidia-smi", "--list-gpus"], check=True)

    test_environment = os.environ.copy()
    test_environment.update(
        {
            "NUMBA_DEVELOPER_MODE": "1",
            "NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING": "1",
            "NUMBA_CAPTURED_ERRORS": "new_style",
            "PYTHONFAULTHANDLER": "1",
            "OMP_TARGET_OFFLOAD": "mandatory",
            "TEST_DEVICE": "gpu",
            "RUN_TARGET": "1",
        }
    )

    for python_version in PYTHON_VERSIONS:
        environment = f"py{python_version.replace('.', '')}"
        python = f"{MINIFORGE_PREFIX}/envs/{environment}/bin/python"
        print(f"Running GPU tests with Python {python_version} and Numba {NUMBA_VERSION}")
        subprocess.run(
            [
                python,
                "-c",
                "from numba.openmp import print_offloading_info; "
                "print_offloading_info()",
            ],
            check=True,
            env=test_environment,
        )
        subprocess.run(
            [
                python,
                "-m",
                "numba.runtests",
                "-v",
                "--",
                "numba.openmp.tests.test_openmp.TestOpenmpTarget",
            ],
            check=True,
            env=test_environment,
        )


@app.local_entrypoint()
def main() -> None:
    test_gpu_wheels.remote()
