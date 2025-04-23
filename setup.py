# setup.py
import os
import numba
import sysconfig
import subprocess
import shutil
import numpy as np
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib


temp_dir = Path("numba/openmp/nrt/numba_src")

bundle_lib = (
    "bundle",
    {
        "sources": [
            "numba/openmp/nrt/init.c",
            f"{temp_dir}/_helpermod.c",
            f"{temp_dir}/cext/utils.c",
            f"{temp_dir}/cext/dictobject.c",
            f"{temp_dir}/cext/listobject.c",
            f"{temp_dir}/core/runtime/_nrt_pythonmod.c",
            f"{temp_dir}/core/runtime/nrt.cpp",
        ],
        "include_dirs": [
            sysconfig.get_paths()["include"],
            np.get_include(),
        ],
    },
)


class BuildStaticNRTBundle(build_clib):
    def finalize_options(self):
        super().finalize_options()
        # Copy numba tree installation to the build directory for building the
        # static library using relative paths.
        numba_dir = numba.__path__[0]
        shutil.copytree(
            numba_dir,
            temp_dir,
            ignore=shutil.ignore_patterns(
                "*.py",
                "*.pyc",
                "*.so",
                "*.dylib",
                "__pycache__",
            ),
            dirs_exist_ok=True,
        )

        self.build_clib = "numba/openmp/libs"

    def run(self):
        super().run()

        # Clean up files after build is completed.
        shutil.rmtree(temp_dir, ignore_errors=True)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir):
        # Don't invoke the original build_ext for this special extension.
        super().__init__(name, sources=[])
        self.sourcedir = sourcedir


class BuildIntrinsicsOpenMPPass(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_cmake(ext)
                return
        super().run()

    def build_cmake(self, ext):
        # Delete build directory if it exists to avoid errors with stale
        # CMakeCache.txt leftovers.
        shutil.rmtree(self.build_temp, ignore_errors=True)

        subprocess.run(
            [
                "cmake",
                "-S",
                ext.sourcedir,
                "-B",
                self.build_temp,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_INSTALL_PREFIX=numba/openmp/libs",
            ],
            check=True,
        )

        subprocess.run(["cmake", "--build", self.build_temp, "-j"], check=True)
        subprocess.run(
            ["cmake", "--install", self.build_temp],
            check=True,
        )


setup(
    libraries=[bundle_lib],
    ext_modules=[CMakeExtension("libIntrinsicsOpenMP", "numba/openmp/pass")],
    cmdclass={
        "build_clib": BuildStaticNRTBundle,
        "build_ext": BuildIntrinsicsOpenMPPass,
    },
)
