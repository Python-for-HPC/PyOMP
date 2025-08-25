#!/bin/bash

set -e

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
export NUMBA_CAPTURED_ERRORS="new_style"
export PYTHONFAULTHANDLER=1

# Disable NumPy dispatching to AVX512_SKX feature extensions if the chip is
# reported to support the feature and NumPy >= 1.22 as this results in the use
# of low accuracy SVML libm replacements in ufunc loops.
_NPY_CMD='from numba.misc import numba_sysinfo;\
          sysinfo=numba_sysinfo.get_sysinfo();\
          print(sysinfo["NumPy AVX512_SKX detected"] and
                sysinfo["NumPy Version"]>="1.22")'
NUMPY_DETECTS_AVX512_SKX_NP_GT_122=$(python -c "$_NPY_CMD")
echo "NumPy >= 1.22 with AVX512_SKX detected: $NUMPY_DETECTS_AVX512_SKX_NP_GT_122"

if [[ "$NUMPY_DETECTS_AVX512_SKX_NP_GT_122" == "True" ]]; then
    export NPY_DISABLE_CPU_FEATURES="AVX512_SKX"
fi

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
  # Test if catchsegv exists, not by default in recent libc.
  if catchsegv --version; then
    SEGVCATCH=catchsegv
  else
    SEGVCATCH=""
  fi
elif [[ "$unamestr" == 'Darwin' ]]; then
  SEGVCATCH=""
else
  echo Error
fi

# Run OpenMP tests in a single-process since they use multiple cores by
# multi-threading. Using multiple processes for testing will very probably slow
# things down.
# XXX: Using -m $TEST_NPROCS, even if with 1 process, hangs on github runners
# when running the full testsuite, while individual tests pass.  This requires
# more investigation. Some observations: 1) running the full test suite creates
# new threads for each region, the old ones are blocked in a futex for
# destruction, 2) it is possible that in small github runners this starves cpu
# time, 3) there may be implications with "-m 1" vs. no flag on how the runtime
# library is inited/de-inited.

echo "=> Run OpenMP CPU parallelism tests"
echo "=> Running: TEST_DEVICES=0 RUN_TARGET=0 $SEGVCATCH python -m numba.runtests -v -- numba.openmp.tests.test_openmp"
# TODO: remove requiring the unused TEST_DEVICES.
TEST_DEVICES=0 RUN_TARGET=0 $SEGVCATCH python -m numba.runtests -v -- numba.openmp.tests.test_openmp 2>&1

echo "=> Run OpenMP offloading tests on CPU (device 1)"
echo "=> Running: TEST_DEVICES=1 RUN_TARGET=1 $SEGVCATCH python -m numba.runtests -v -- numba.openmp.tests.test_openmp.TestOpenmpTarget"
OMP_TARGET_OFFLOAD=mandatory TEST_DEVICES=1 RUN_TARGET=1 $SEGVCATCH python -m numba.runtests -v -- numba.openmp.tests.test_openmp.TestOpenmpTarget 2>&1
if nvidia-smi --list-gpus; then
  echo "=> Found NVIDIA GPU, Run OpenMP offloading tests on GPU (device 0)"
  echo "=> Running: TEST_DEVICES=0 RUN_TARGET=1 $SEGVCATCH python -m numba.runtests -v -- numba.openmp.tests.test_openmp.TestOpenmpTarget"
  OMP_TARGET_OFFLOAD=mandatory TEST_DEVICES=0 RUN_TARGET=1 $SEGVCATCH python -m numba.runtests -v -- numba.openmp.tests.test_openmp.TestOpenmpTarget 2>&1
fi
