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

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None


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


if _bdist_wheel:

    class CustomBdistWheel(_bdist_wheel):
        def run(self):
            # Ensure all build steps are run before bdist_wheel
            self.run_command("build_ext")
            super().run()
else:
    CustomBdistWheel = None


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
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            ]
            + ext.cmake_args
            + extra_cmake_args
        )
        subprocess.run(cfg, check=True, stdin=subprocess.DEVNULL)

        subprocess.run(
            ["cmake", "--build", build_dir, "-j"], check=True, stdin=subprocess.DEVNULL
        )
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
        return args


def _prepare_source_openmp(sha256=None):
    LLVM_VERSION = os.environ.get("LLVM_VERSION", None)
    assert LLVM_VERSION is not None, "LLVM_VERSION environment variable must be set."
    url = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}/openmp-{LLVM_VERSION}.src.tar.xz"

    tmp = Path("_downloads/libomp") / f"openmp-{LLVM_VERSION}.tar.gz"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    # Download the source tarball if it does not exist.
    if not tmp.exists():
        print(f"download openmp version {LLVM_VERSION} url:", url)
        with urllib.request.urlopen(url) as r:
            with tmp.open("wb") as f:
                f.write(r.read())

    # Extract only the major version.
    llvm_major_version = tuple(map(int, LLVM_VERSION.split(".")))[0]
    # For LLVM versions > 14, we also need to download CMake files.
    if llvm_major_version > 14:
        cmake_url = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}/cmake-{LLVM_VERSION}.src.tar.xz"
        cmake_file = Path("_downloads/libomp") / f"cmake-{LLVM_VERSION}.tar.gz"
        if not cmake_file.exists():
            with urllib.request.urlopen(cmake_url) as r:
                with cmake_file.open("wb") as f:
                    f.write(r.read())
        with tarfile.open(cmake_file) as tf:
            tf.extractall(cmake_file.parent)
            src = cmake_file.parent / tf.getnames()[0]
            dst = cmake_file.parent / "cmake"
            if not dst.exists():
                src.rename(dst)

    if sha256:
        import hashlib

        hasher = hashlib.sha256()
        with tmp.open("rb") as f:
            hasher.update(f.read())
        if hasher.hexdigest() != sha256:
            raise ValueError(f"SHA256 mismatch for {url}")

    with tarfile.open(tmp) as tf:
        # We assume the tarball contains a single directory with the source files.
        sourcedir = tmp.parent / tf.getnames()[0]
        tf.extractall(tmp.parent)

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

    return sourcedir


setup(
    ext_modules=[
        CMakeExtension("pass", sourcedir="src/numba/openmp/libs/pass"),
        CMakeExtension(
            "libomp",
            sourcedir=_prepare_source_openmp(),
            cmake_args=[
                "-DLIBOMP_OMPD_SUPPORT=OFF",
                "-DLIBOMP_OMPT_SUPPORT=OFF",
                "-DCMAKE_INSTALL_LIBDIR=lib",
            ],
        ),
    ],
    cmdclass={
        "clean": CleanCommand,
        "build_ext": BuildCMakeExt,
        **({"bdist_wheel": CustomBdistWheel} if CustomBdistWheel else {}),
    },
)
