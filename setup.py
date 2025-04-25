import numba
import sysconfig
import subprocess
import shutil
import numpy as np
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib


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
                f"-DCMAKE_INSTALL_PREFIX={self.build_lib}/numba/openmp/libs",
            ],
            check=True,
        )

        subprocess.run(["cmake", "--build", self.build_temp, "-j"], check=True)
        subprocess.run(
            ["cmake", "--install", self.build_temp],
            check=True,
        )


setup(
    libraries=[nrt_static],
    ext_modules=[CMakeExtension("libIntrinsicsOpenMP", "numba/openmp/libs/pass")],
    cmdclass={
        "build_clib": BuildStaticNRT,
        "build_ext": BuildIntrinsicsOpenMPPass,
    },
)
