# setup.py
import os
import numba
import sysconfig
import numpy as np
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib


numba_dir = os.path.dirname(numba.__file__)
bundle_lib = (
    "bundle",
    {
        "sources": [
            "numba/openmp/nrt/init.c",
            f"{numba_dir}/_helpermod.c",
            f"{numba_dir}/cext/utils.c",
            f"{numba_dir}/cext/dictobject.c",
            f"{numba_dir}/cext/listobject.c",
            f"{numba_dir}/core/runtime/_nrt_pythonmod.c",
            f"{numba_dir}/core/runtime/nrt.cpp",
        ],
        "include_dirs": [
            sysconfig.get_paths()["include"],
            np.get_include(),
        ],
    },
)


class BuildStaticBundle(build_clib):
    def finalize_options(self):
        super().finalize_options()
        self.build_temp = (Path("numba/openmp/nrt") / self.build_temp).absolute()
        self.build_temp.mkdir(parents=True, exist_ok=True)
        self.build_temp = str(self.build_temp)
        self.build_clib = str(Path("numba/openmp/libs").absolute())


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])
        self.sourcedir = sourcedir


class BuildPass(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_cmake(ext)
                return
        super().run()

    def build_cmake(self, ext):
        build_dir = (Path(ext.sourcedir) / self.build_temp).absolute()
        subprocess.run(
            [
                "cmake",
                "-S",
                ext.sourcedir,
                "-B",
                build_dir,
                "--install-prefix",
                Path("numba/openmp/libs").absolute(),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
        )

        subprocess.run(["cmake", "--build", build_dir, "-j"], check=True)
        subprocess.run(
            ["cmake", "--install", build_dir],
            check=True,
        )


setup(
    libraries=[bundle_lib],
    ext_modules=[CMakeExtension("libIntrinsicsOpenMP", "numba/openmp/pass")],
    cmdclass={"build_clib": BuildStaticBundle, "build_ext": BuildPass},
)
