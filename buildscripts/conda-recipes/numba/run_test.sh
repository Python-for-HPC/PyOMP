#!/bin/bash

set -e

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
export NUMBA_CAPTURED_ERRORS="new_style"
export PYTHONFAULTHANDLER=1
# Required OpenMP test env var (for offloading).
export TEST_DEVICES=0

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

# limit CPUs in use on PPC64LE, fork() issues
# occur on high core count systems
archstr=`uname -m`
if [[ "$archstr" == 'ppc64le' ]]; then
    TEST_NPROCS=16
fi

# Check Numba executable is there
numba -h

# run system info tool
numba -s

# Check test discovery works
python -m numba.tests.test_runtests

# Disable tests for package building.
exit 0

if nvidia-smi --list-gpus; then
  echo "Found NVIDIA GPU, enable OpenMP offloading tests"
  export RUN_TARGET=1
else
  echo "Missing NVIDIA GPU, disable OpenMP offloading tests"
  export RUN_TARGET=0
fi

# Run the whole test suite
# Test only openmp for brevity. We may want to enable the full numba tests,
# which include openmp, on larger runners.
TESTS_TO_RUN="numba.tests.test_openmp"
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

echo "Running: $SEGVCATCH python -m numba.runtests -v -- $TESTS_TO_RUN"
$SEGVCATCH python -m numba.runtests -v -- $TESTS_TO_RUN
