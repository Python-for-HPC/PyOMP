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
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib

OPENMP_URL = "https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.6/openmp-14.0.6.src.tar.xz"
OPENMP_SHA256 = "4f731ff202add030d9d68d4c6daabd91d3aeed9812e6a5b4968815cfdff0eb1f"

nrt_static = (
    "nrt_static",
    {
        # We extend those sources with the ones from the numba tree.
        "sources": [
            "numba/openmp/libs/nrt/init.c",
        ],
        "include_dirs": [
            sysconfig.get_paths()["include"],
            np.get_include(),
        ],
    },
)


class BuildStaticNRT(build_clib):
    def finalize_options(self):
        super().finalize_options()
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

        libname, build_info = self.libraries[0]
        if libname != "nrt_static":
            raise Exception("Expected library name 'nrt_static'")
        if len(self.libraries) != 1:
            raise Exception("Expected only the `nrt_static' library in the list")

        sources = build_info["sources"]
        sources.extend(
            [
                f"{self.build_temp}/numba_src/_helpermod.c",
                f"{self.build_temp}/numba_src/cext/utils.c",
                f"{self.build_temp}/numba_src/cext/dictobject.c",
                f"{self.build_temp}/numba_src/cext/listobject.c",
                f"{self.build_temp}/numba_src/core/runtime/_nrt_pythonmod.c",
                f"{self.build_temp}/numba_src/core/runtime/nrt.cpp",
            ]
        )

        # Get build_lib directory from the 'build' command.
        build_cmd = self.get_finalized_command("build")
        # Build the static library in the wheel output build directory.
        self.build_clib = f"{build_cmd.build_lib}/numba/openmp/libs"


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

        extra_cmake_args = self._env_toolchain_args()
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
        subprocess.run(
            cfg,
            check=True,
        )

        subprocess.run(["cmake", "--build", build_dir, "-j"], check=True)
        subprocess.run(
            ["cmake", "--install", build_dir],
            check=True,
        )

        # Remove symlinks in the lib directory to avoid issues with creating the
        # wheel.
        for file in lib_dir.rglob("*"):
            if file.is_symlink():
                file.unlink()

    def _env_toolchain_args(self) -> list[str]:
        args = []
        # macOS archs/deployment target (cibuildwheel exposes these)
        if archs := os.environ.get("ARCHS"):
            args += [f"-DCMAKE_OSX_ARCHITECTURES={archs}"]
        if minver := os.environ.get("MACOSX_DEPLOYMENT_TARGET"):
            args += [f"-DCMAKE_OSX_DEPLOYMENT_TARGET={minver}"]

        # Generic toolchain env
        for var, cm in [
            ("CC", "-DCMAKE_C_COMPILER="),
            ("CXX", "-DCMAKE_CXX_COMPILER="),
            ("AR", "-DCMAKE_AR="),
            ("RANLIB", "-DCMAKE_RANLIB="),
            ("NM", "-DCMAKE_NM="),
        ]:
            if os.environ.get(var):
                args.append(cm + os.environ[var])

        # Flags
        if os.environ.get("CFLAGS"):
            args.append(f"-DCMAKE_C_FLAGS={os.environ['CFLAGS']}")
        if os.environ.get("CXXFLAGS"):
            args.append(f"-DCMAKE_CXX_FLAGS={os.environ['CXXFLAGS']}")
        if os.environ.get("LDFLAGS"):
            args.append(f"-DCMAKE_EXE_LINKER_FLAGS={os.environ['LDFLAGS']}")
        return args


setup(
    libraries=[nrt_static],
    ext_modules=[
        CMakeExtension("pass", sourcedir="numba/openmp/libs/pass"),
        CMakeExtension(
            "libomp",
            url=OPENMP_URL,
            sha256=OPENMP_SHA256,
        ),
    ],
    cmdclass={
        "build_clib": BuildStaticNRT,
        "build_ext": BuildCMakeExt,
    },
)
