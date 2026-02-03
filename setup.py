from pathlib import Path
import subprocess
import shutil
import tarfile
import urllib
import sys
import os
from setuptools import setup, Extension
from setuptools import Command
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path in ["build", "dist", "src/numba/openmp/libs"]:
            shutil.rmtree(path, ignore_errors=True)
        for egg_info in Path("src").rglob("*.egg-info"):
            shutil.rmtree(egg_info, ignore_errors=True)


class CustomBdistWheel(bdist_wheel):
    def run(self):
        # Ensure all build steps are run before bdist_wheel
        self.run_command("build_ext")
        super().run()


class CMakeExtension(Extension):
    def __init__(self, name, *, sourcedir=None, url=None, sha256=None, cmake_args=[]):
        # Don't invoke the original build_ext for this special extension.
        super().__init__(name, sources=[])
        if sourcedir and url:
            raise ValueError(
                "CMakeExtension should have either a sourcedir or a url, not both."
            )
        self.sourcedir = sourcedir
        self.url = url
        self.sha256 = sha256
        self.cmake_args = cmake_args


class BuildCMakeExt(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self._build_cmake(ext)
            else:
                super().run()

    def _build_cmake(self, ext: CMakeExtension):
        print("Build CMake extension:", ext.name)
        # Delete build directory if it exists to avoid errors with stale
        # CMakeCache.txt leftovers.
        build_dir = Path(self.build_temp) / ext.name
        shutil.rmtree(build_dir, ignore_errors=True)
        build_dir.mkdir(parents=True, exist_ok=True)

        lib_dir = Path(
            self.get_finalized_command("build_py").get_package_dir("numba.openmp.libs")
        )

        extra_cmake_args = self._env_toolchain_args(ext)
        # Set RPATH.
        if sys.platform.startswith("linux"):
            extra_cmake_args.append(r"-DCMAKE_INSTALL_RPATH=$ORIGIN")
        elif sys.platform == "darwin":
            extra_cmake_args.append(r"-DCMAKE_INSTALL_RPATH=@loader_path")

        install_dir = Path(lib_dir) / ext.name
        install_dir.mkdir(parents=True, exist_ok=True)
        cfg = (
            [
                "cmake",
                "-S",
                ext.sourcedir,
                "-B",
                build_dir,
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            ]
            + ext.cmake_args
            + extra_cmake_args
        )
        print("Configure cmake with args:", cfg)
        subprocess.run(cfg, check=True, stdin=subprocess.DEVNULL)

        print("Build at dir ", build_dir)
        subprocess.run(
            ["cmake", "--build", build_dir, "-j"], check=True, stdin=subprocess.DEVNULL
        )
        print("Install at dir ", install_dir)
        subprocess.run(
            ["cmake", "--install", build_dir], check=True, stdin=subprocess.DEVNULL
        )

        # Remove unnecessary files after installing libomp.
        if ext.name == "libomp":
            # Remove include directory after install.
            include_dir = install_dir / "include"
            if include_dir.exists():
                shutil.rmtree(include_dir)
            # Remove cmake directory after install.
            include_dir = install_dir / "lib/cmake"
            if include_dir.exists():
                shutil.rmtree(include_dir)

    def _env_toolchain_args(self, ext):
        args = []
        # Forward LLVM_DIR if provided.
        if os.environ.get("LLVM_DIR"):
            args.append(f"-DLLVM_DIR={os.environ['LLVM_DIR']}")
        # Forward CC, CXX if provided.
        if os.environ.get("CC"):
            args.append(f"-DCMAKE_C_COMPILER={os.environ['CC']}")
        if os.environ.get("CXX"):
            args.append(f"-DCMAKE_CXX_COMPILER={os.environ['CXX']}")
        return args


def _prepare_source_openmp(sha256=None):
    LLVM_VERSION = os.environ.get("LLVM_VERSION", None)
    assert LLVM_VERSION is not None, "LLVM_VERSION environment variable must be set."
    url = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}/llvm-project-{LLVM_VERSION}.src.tar.xz"

    tmp = Path("_downloads/libomp") / f"llvm-project-{LLVM_VERSION}.tar.gz"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    # Download the source tarball if it does not exist.
    if not tmp.exists():
        print(f"Downloading llvm-project version {LLVM_VERSION} url:", url)
        with urllib.request.urlopen(url) as r:
            with tmp.open("wb") as f:
                f.write(r.read())
    else:
        print(f"Using downloaded llvm-project at {tmp}")

    if sha256:
        import hashlib

        hasher = hashlib.sha256()
        with tmp.open("rb") as f:
            hasher.update(f.read())
        if hasher.hexdigest() != sha256:
            raise ValueError(f"SHA256 mismatch for {url}")

    print("Extracting llvm-project...")
    with tarfile.open(tmp) as tf:
        # The root dir llvm-project-20.1.8.src
        root_name = tf.getnames()[0]

        # Extract only needed subdirectories
        members = [
            m
            for m in tf.getmembers()
            if m.name.startswith(f"{root_name}/openmp/")
            or m.name.startswith(f"{root_name}/offload/")
            or m.name.startswith(f"{root_name}/runtimes/")
            or m.name.startswith(f"{root_name}/cmake/")
            or m.name.startswith(f"{root_name}/llvm/cmake/")
            or m.name.startswith(f"{root_name}/llvm/utils/")
            or m.name.startswith(f"{root_name}/libc/")
        ]

        parentdir = tmp.parent
        tf.extractall(path=parentdir, members=members, filter="data")
        sourcedir = parentdir / root_name
        print("Extracted llvm-project to:", sourcedir)

    print("Applying patches to llvm-project...")
    for patch in (
        Path(f"src/numba/openmp/libs/libomp/patches/{LLVM_VERSION}")
        .absolute()
        .glob("*.patch")
    ):
        print("applying patch", patch)
        subprocess.run(
            ["patch", "-p1", "-i", str(patch)],
            cwd=sourcedir,
            check=True,
            stdin=subprocess.DEVNULL,
        )

    return f"{sourcedir}/runtimes"


setup(
    ext_modules=[
        CMakeExtension("pass", sourcedir="src/numba/openmp/libs/pass"),
        CMakeExtension(
            "libomp",
            sourcedir=_prepare_source_openmp(),
            cmake_args=[
                "-DOPENMP_STANDALONE_BUILD=ON",
                "-DLLVM_ENABLE_RUNTIMES=openmp;offload",
                "-DOPENMP_ENABLE_OMPT_TOOLS=OFF",
                # Avoid conflicts in manylinux builds with packaged clang/llvm
                # under /usr/include and its gcc-toolset provided header files.
                "-DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON",
            ],
        ),
    ],
    cmdclass={
        "clean": CleanCommand,
        "build_ext": BuildCMakeExt,
        **({"bdist_wheel": CustomBdistWheel} if CustomBdistWheel else {}),
    },
)
