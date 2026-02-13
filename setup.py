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


class CMakeExtension(Extension):
    def __init__(
        self,
        name,
        *,
        setup=None,
        source_dir=None,
        install_dir=None,
        cmake_args=[],
    ):
        # Don't invoke the original build_ext for this special extension.
        super().__init__(name, sources=[])
        self.setup = setup
        self.source_dir = source_dir
        self.install_dir = install_dir
        self.cmake_args = cmake_args


class BuildCMakeExt(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self._build_cmake(ext)
            else:
                super().run()

    def finalize_options(self):
        super().finalize_options()
        # Create placeholder directories for package-data validation.
        Path("src/numba/openmp/libs/openmp/lib").mkdir(parents=True, exist_ok=True)

    def _build_cmake(self, ext: CMakeExtension):
        print("Build CMake extension:", ext.name)
        # Delete build directory if it exists to avoid errors with stale
        # CMakeCache.txt leftovers.
        build_dir = Path(self.build_temp) / ext.name
        shutil.rmtree(build_dir, ignore_errors=True)
        build_dir.mkdir(parents=True, exist_ok=True)

        if ext.setup:
            ext.setup.setup()

        if self.inplace:
            lib_dir = Path(
                self.get_finalized_command("build_py").get_package_dir(
                    "numba.openmp.libs"
                )
            )
        else:
            lib_dir = Path(self.build_lib) / "numba/openmp/libs"

        extra_cmake_args = self._env_toolchain_args(ext)
        # Set RPATH.
        if sys.platform.startswith("linux"):
            extra_cmake_args.append(r"-DCMAKE_INSTALL_RPATH=$ORIGIN")
        elif sys.platform == "darwin":
            extra_cmake_args.append(r"-DCMAKE_INSTALL_RPATH=@loader_path")

        if ext.install_dir is None:
            install_dir = Path(lib_dir) / ext.name
        else:
            install_dir = Path(lib_dir) / ext.install_dir
        install_dir.mkdir(parents=True, exist_ok=True)

        cfg = (
            [
                "cmake",
                "-S",
                ext.source_dir,
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
        if ext.name.startswith("libomp"):
            # Remove include directory after install.
            include_dir = install_dir / "include"
            if include_dir.exists():
                shutil.rmtree(include_dir)
            # Remove cmake directory after install.
            include_dir = install_dir / "lib/cmake"
            if include_dir.exists():
                shutil.rmtree(include_dir)
        # Remove symlinks in the install directory to avoid copies.
        for file in install_dir.rglob("*"):
            if file.is_symlink():
                file.unlink()

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


class PrepareOpenMP:
    setup_done = False
    LLVM_VERSION = os.environ.get("LLVM_VERSION", None)

    @classmethod
    def setup(cls):
        if not cls.setup_done:
            cls._prepare_source_openmp()
            cls.setup_done = True

    @classmethod
    def get_source_dir(cls):
        return Path(
            f"_downloads/libomp/llvm-project-{cls.LLVM_VERSION}.src/runtimes"
        ).absolute()

    @classmethod
    def _prepare_source_openmp(cls):
        assert cls.LLVM_VERSION is not None, (
            "LLVM_VERSION environment variable must be set."
        )
        url = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{cls.LLVM_VERSION}/llvm-project-{cls.LLVM_VERSION}.src.tar.xz"

        tmp = Path("_downloads/libomp") / f"llvm-project-{cls.LLVM_VERSION}.tar.gz"
        tmp.parent.mkdir(parents=True, exist_ok=True)

        # Download the source tarball if it does not exist.
        if not tmp.exists():
            print(
                f"Downloading llvm-project version {cls.LLVM_VERSION} url:",
                url,
                flush=True,
            )
            with urllib.request.urlopen(url) as r:
                with tmp.open("wb") as f:
                    f.write(r.read())
        else:
            print(f"Using downloaded llvm-project at {tmp}", flush=True)

        print("Extracting llvm-project...", flush=True)
        root_dir = Path(f"_downloads/libomp/llvm-project-{cls.LLVM_VERSION}.src")
        if not root_dir.exists():
            with tarfile.open(tmp) as tf:
                # Extract only needed subdirectories
                root_name = f"llvm-project-{cls.LLVM_VERSION}.src"

                # Use prefix tuple + any() for faster membership testing
                prefixes = (
                    f"{root_name}/openmp/",
                    f"{root_name}/offload/",
                    f"{root_name}/runtimes/",
                    f"{root_name}/cmake/",
                    f"{root_name}/llvm/cmake/",
                    f"{root_name}/llvm/utils/",
                    f"{root_name}/libc/",
                )

                include_members = [
                    m
                    for m in tf.getmembers()
                    if any(m.name.startswith(p) for p in prefixes)
                ]

                parentdir = tmp.parent
                # Base arguments for extractall.
                kwargs = {"path": parentdir, "members": include_members}

                # Check if data filter is available.
                if hasattr(tarfile, "data_filter"):
                    # If this exists, the 'filter' argument is guaranteed to work
                    kwargs["filter"] = "data"

                tf.extractall(**kwargs)

            print("Applying patches to llvm-project...", flush=True)
            for patch in sorted(
                Path(f"src/numba/openmp/libs/openmp/patches/{cls.LLVM_VERSION}")
                .absolute()
                .glob("*.patch")
            ):
                print("applying patch", patch, flush=True)
                subprocess.run(
                    ["patch", "-p1", "-N", "-i", str(patch)],
                    cwd=root_dir,
                    check=True,
                    stdin=subprocess.DEVNULL,
                )


def _check_true(env_var):
    val = os.environ.get(env_var, "0")
    return val.lower() in ("1", "true", "yes", "on")


# Build extensions: always include 'pass', conditionally include 'openmp'
# libraries.
ext_modules = [CMakeExtension("pass", source_dir="src/numba/openmp/libs/pass")]


# Optionally enable bundled libomp build via ENABLE_BUNDLED_LIBOMP=1.  We want
# to avoid bundling for conda builds to avoid duplicate OpenMP runtime conflicts
# (e.g., numba 0.62+ and libopenblas already require llvm-openmp).
if _check_true("ENABLE_BUNDLED_LIBOMP"):
    ext_modules.append(
        CMakeExtension(
            "libomp",
            setup=PrepareOpenMP,
            source_dir=PrepareOpenMP.get_source_dir(),
            install_dir="openmp",
            cmake_args=[
                "-DOPENMP_STANDALONE_BUILD=ON",
                "-DLLVM_ENABLE_RUNTIMES=openmp",
                "-DLIBOMP_OMPD_SUPPORT=OFF",
                "-DOPENMP_ENABLE_OMPT_TOOLS=OFF",
                # Avoid conflicts in manylinux builds with packaged clang/llvm
                # under /usr/include and its gcc-toolset provided header files.
                "-DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON",
            ],
        )
    )

# Optionally enable bundled libomptarget build via ENABLE_BUNDLED_LIBOMPTARGET=1.
# We avoid building and bundling for unsupported platforms.
if _check_true("ENABLE_BUNDLED_LIBOMPTARGET"):
    ext_modules.append(
        CMakeExtension(
            "libomptarget",
            setup=PrepareOpenMP,
            source_dir=PrepareOpenMP.get_source_dir(),
            install_dir="openmp",
            cmake_args=[
                "-DOPENMP_STANDALONE_BUILD=ON",
                "-DLLVM_ENABLE_RUNTIMES=offload",
                # Avoid conflicts in manylinux builds with packaged clang/llvm
                # under /usr/include and its gcc-toolset provided header files.
                "-DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON",
            ],
        )
    )


setup(
    ext_modules=ext_modules,
    cmdclass={
        "clean": CleanCommand,
        "build_ext": BuildCMakeExt,
    },
)
