import os

import numpy as np
import pytest

from numba.openmp import njit
from numba.openmp import openmp_context as openmp


@njit
def parallel_max_float64(array):
    result = -np.inf
    with openmp("parallel for reduction(max:result)"):
        for j in range(array.size):
            if array[j] > result:
                result = array[j]
    return result


@njit
def parallel_max_int64(array):
    result = np.iinfo(np.int64).min
    with openmp("parallel for reduction(max:result)"):
        for j in range(array.size):
            if array[j] > result:
                result = array[j]
    return result


@njit
def parallel_max_uint64(array):
    result = np.uint64(0)
    with openmp("parallel for reduction(max:result)"):
        for j in range(array.size):
            if array[j] > result:
                result = array[j]
    return result


@njit
def parallel_max_with_initial_value(array):
    result = 20.0
    with openmp("parallel for reduction(max:result)"):
        for j in range(array.size):
            if array[j] > result:
                result = array[j]
    return result


@njit
def parallel_min_float64(array):
    result = np.inf
    with openmp("parallel for reduction(min:result)"):
        for j in range(array.size):
            if array[j] < result:
                result = array[j]
    return result


@njit
def parallel_min_int64(array):
    result = np.iinfo(np.int64).max
    with openmp("parallel for reduction(min:result)"):
        for j in range(array.size):
            if array[j] < result:
                result = array[j]
    return result


@njit
def parallel_min_uint64(array):
    result = np.iinfo(np.uint64).max
    with openmp("parallel for reduction(min:result)"):
        for j in range(array.size):
            if array[j] < result:
                result = array[j]
    return result


@njit
def target_max_float64(array):
    result = -np.inf
    with openmp("target map(to:array) map(tofrom:result)"):
        with openmp("loop reduction(max:result)"):
            for j in range(array.size):
                if array[j] > result:
                    result = array[j]
    return result

def test_parallel_max_float64_all_positive():
    array = np.array([4.0, 7.0, 1.5, 9.0, 2.0], dtype=np.float64)
    assert parallel_max_float64(array) == 9.0

def test_parallel_max_float64_all_negative():
    array = np.array([-4.0, -7.0, -1.5, -9.0, -2.0], dtype=np.float64)
    assert parallel_max_float64(array) == -1.5


def test_parallel_max_float64_non_power_of_two():
    array = np.array([-3.0, 7.0, 1.0, 7.0, -2.0, 4.0, 6.0], dtype=np.float64)
    assert parallel_max_float64(array) == 7.0


def test_parallel_max_float64_single_element():
    array = np.array([-12.5], dtype=np.float64)
    assert parallel_max_float64(array) == -12.5


def test_parallel_max_preserve_original_value():
    array = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    assert parallel_max_with_initial_value(array) == 20.0


def test_parallel_max_int64_negative():
    array = np.array(
        [np.iinfo(np.int64).min, -100, -2, -50],
        dtype=np.int64,
    )
    assert parallel_max_int64(array) == -2


def test_parallel_max_uint64_above_signed_range():
    array = np.array(
        [1, 2**63 + 5, np.iinfo(np.uint64).max],
        dtype=np.uint64,
    )
    assert parallel_max_uint64(array) == np.iinfo(np.uint64).max


def test_parallel_max_matche_numpy():
    rng = np.random.default_rng(12345)
    array = rng.standard_normal(100_003).astype(np.float64)
    assert parallel_max_float64(array) == np.max(array)


def test_parallel_min_float64_all_positive():
    array = np.array([4.0, 7.0, 1.5, 9.0, 2.0], dtype=np.float64)
    assert parallel_min_float64(array) == 1.5

def test_parallel_min_float64_all_negative():
    array = np.array([-4.0, -7.0, -1.5, -9.0, -2.0], dtype=np.float64)
    assert parallel_min_float64(array) == -9.0


def test_parallel_min_int64_extreme():
    array = np.array(
        [np.iinfo(np.int64).max, -100, np.iinfo(np.int64).min, -50],
        dtype=np.int64,
    )
    assert parallel_min_int64(array) == np.iinfo(np.int64).min


def test_parallel_min_uint64_above_signed_range():
    array = np.array(
        [np.iinfo(np.uint64).max, 2**63 + 5, 2**63 + 100],
        dtype=np.uint64,
    )
    assert parallel_min_uint64(array) == 2**63 + 5


def _target_offload_requested():
    return os.environ.get("OMP_TARGET_OFFLOAD", "").upper() == "MANDATORY"

@pytest.mark.skipif(
    not _target_offload_requested(),
    reason="need OMP_TARGET_OFFLOAD=MANDATORY",
)
def test_target_max_float64_all_positive():
    array = np.array([4.0, 7.0, 1.5, 9.0, 2.0], dtype=np.float64)
    assert target_max_float64(array) == 9.0

@pytest.mark.skipif(
    not _target_offload_requested(),
    reason="need OMP_TARGET_OFFLOAD=MANDATORY",
)
def test_target_max_float64_all_negative():
    array = np.array([-4.0, -7.0, -1.5, -9.0, -2.0], dtype=np.float64)
    assert target_max_float64(array) == -1.5


@pytest.mark.skipif(
    not _target_offload_requested(),
    reason="need OMP_TARGET_OFFLOAD=MANDATORY",
)
def test_target_max_float64_non_power_of_two():
    array = np.array([-3.0, 7.0, 1.0, 7.0, -2.0, 4.0, 6.0], dtype=np.float64)
    assert target_max_float64(array) == 7.0
