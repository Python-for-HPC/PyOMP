from pathlib import Path
import numba
import sysconfig
import subprocess
import shutil
import numpy as np
import tarfile
import urllib
import sys
import os
from setuptools import setup, Extension
from setuptools import Command
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib
from setuptools.command.sdist import sdist as _sdist

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None

OPENMP_URL = "https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.6/openmp-14.0.6.src.tar.xz"
OPENMP_SHA256 = "4f731ff202add030d9d68d4c6daabd91d3aeed9812e6a5b4968815cfdff0eb1f"

nrt_static = (
    "nrt_static",
    {
        # We extend those sources with the ones from the numba tree.
        "sources": [
            "src/libs/nrt/init.c",
        ],
        "include_dirs": [
            sysconfig.get_paths()["include"],
            np.get_include(),
        ],
    },
)


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


class CustomSdist(_sdist):
    def run(self):
        # Ensure all build steps are run before sdist
        self.run_command("build_clib")
        self.run_command("build_ext")
        super().run()


if _bdist_wheel:

    class CustomBdistWheel(_bdist_wheel):
        def run(self):
            # Ensure all build steps are run before bdist_wheel
            self.run_command("build_clib")
            self.run_command("build_ext")
            super().run()
else:
    CustomBdistWheel = None


class BuildStaticNRT(build_clib):
    def run(self):
        super().run()
        # Find the built static library
        libname = "libnrt_static.a"
        build_lib_path = Path(self.build_clib) / libname
        # Destination: your package directory
        dest = Path("src/numba/openmp/libs/nrt") / libname
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build_lib_path, dest)

    def build_libraries(self, libraries):
        for libname, build_info in libraries:
            if libname == "nrt_static":
                self._prepare_nrt_static(build_info)

        super().build_libraries(libraries)

    def _prepare_nrt_static(self, build_info):
        # Copy numba tree installation to the temp directory for building the
        # static library using relative paths.
        numba_dir = numba.__path__[0]
        shutil.copytree(
            numba_dir,
            f"{self.build_temp}/numba_src",
            ignore=shutil.ignore_patterns(
                "*.py",
                "*.pyc",
                "*.so",
                "*.dylib",
                "__pycache__",
            ),
            dirs_exist_ok=True,
        )

        sources = set(build_info["sources"])
        sources.update(
            [
                f"{self.build_temp}/numba_src/_helpermod.c",
                f"{self.build_temp}/numba_src/cext/utils.c",
                f"{self.build_temp}/numba_src/cext/dictobject.c",
                f"{self.build_temp}/numba_src/cext/listobject.c",
                f"{self.build_temp}/numba_src/core/runtime/_nrt_pythonmod.c",
                f"{self.build_temp}/numba_src/core/runtime/nrt.cpp",
            ]
        )
        build_info["sources"] = list(sources)


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
                self._prepare_source(ext)
                self._build_cmake(ext)
            else:
                super().run()

    def _prepare_source(self, ext):
        if ext.sourcedir:
            return

        tmp = Path("_downloads") / f"{ext.name}" / "src.tar.gz"
        tmp.parent.mkdir(parents=True, exist_ok=True)

        # Download the source tarball if it does not exist.
        if not tmp.exists():
            with urllib.request.urlopen(ext.url) as r:
                with tmp.open("wb") as f:
                    f.write(r.read())

        if ext.sha256:
            import hashlib

            sha256 = hashlib.sha256()
            with tmp.open("rb") as f:
                sha256.update(f.read())
            if sha256.hexdigest() != ext.sha256:
                raise ValueError(f"SHA256 mismatch for {ext.url}")

        with tarfile.open(tmp) as tf:
            # We assume the tarball contains a single directory with the source files.
            ext.sourcedir = tmp.parent / tf.getnames()[0]
            tf.extractall(tmp.parent)

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

        # Remove symlinks in the install directory to avoid issues with creating
        # the wheel.
        for file in install_dir.rglob("*"):
            if file.is_symlink():
                file.unlink()

    def _env_toolchain_args(self, ext) -> list[str]:
        args = []
        # Set macOS deployment target and architectures if provided.
        if archs := os.environ.get("ARCHS"):
            args += [f"-DCMAKE_OSX_ARCHITECTURES={archs}"]
        if minver := os.environ.get("MACOSX_DEPLOYMENT_TARGET"):
            args += [f"-DCMAKE_OSX_DEPLOYMENT_TARGET={minver}"]

        # Forward compiler tool environment variables.
        for var, cm in [
            ("CC", "-DCMAKE_C_COMPILER="),
            ("CXX", "-DCMAKE_CXX_COMPILER="),
            ("AR", "-DCMAKE_AR="),
            ("RANLIB", "-DCMAKE_RANLIB="),
            ("NM", "-DCMAKE_NM="),
        ]:
            if os.environ.get(var):
                args.append(cm + os.environ[var])

        # Forward compilation and linking flags.
        if os.environ.get("CFLAGS"):
            args.append(f"-DCMAKE_C_FLAGS={os.environ['CFLAGS']}")
        if os.environ.get("CXXFLAGS"):
            args.append(f"-DCMAKE_CXX_FLAGS={os.environ['CXXFLAGS']}")
        if os.environ.get("LDFLAGS"):
            args.append(f"-DCMAKE_EXE_LINKER_FLAGS={os.environ['LDFLAGS']}")

        # Set LLVM_DIR and CLANG_TOOL if provided.
        if os.environ.get("LLVM_DIR"):
            args.append(f"-DLLVM_DIR={os.environ['LLVM_DIR']}")
        if ext.name == "libomp":
            # CLANG_TOOL is used by libomp to find clang for generating the OpenMP
            # device runtime bitcodes.
            if os.environ.get("CLANG_TOOL"):
                args.append(f"-DCLANG_TOOL={os.environ['CLANG_TOOL']}")
        return args


setup(
    libraries=[nrt_static],
    ext_modules=[
        CMakeExtension("pass", sourcedir="src/libs/pass"),
        CMakeExtension(
            "libomp",
            url=OPENMP_URL,
            sha256=OPENMP_SHA256,
        ),
    ],
    cmdclass={
        "clean": CleanCommand,
        "build_clib": BuildStaticNRT,
        "build_ext": BuildCMakeExt,
        "sdist": CustomSdist,
        **({"bdist_wheel": CustomBdistWheel} if CustomBdistWheel else {}),
    },
)
