import contextlib
import math
import numbers
import os
import platform
import numpy as np

from numba.core import (
    compiler,
)

from numba.core.compiler import (
    Flags,
)
from numba.tests.support import (
    TestCase,
    override_env_config,
    linux_only,
)

import numba.openmp
from numba.openmp import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import (
    omp_set_num_threads,
    omp_get_thread_num,
    omp_get_num_threads,
    omp_get_wtime,
    omp_set_nested,
    omp_set_max_active_levels,
    omp_set_dynamic,
    omp_get_max_active_levels,
    omp_get_max_threads,
    omp_get_num_procs,
    UnspecifiedVarInDefaultNone,
    NonconstantOpenmpSpecification,
    NonStringOpenmpSpecification,
    omp_get_thread_limit,
    ParallelForExtraCode,
    ParallelForWrongLoopCount,
    omp_in_parallel,
    omp_get_level,
    omp_get_active_level,
    omp_get_team_size,
    omp_get_ancestor_thread_num,
    omp_get_team_num,
    omp_get_num_teams,
    omp_in_final,
    omp_shared_array,
)
import cmath
import unittest

# NOTE: Each OpenMP test class is run in separate subprocess, this is to reduce
# memory pressure in CI settings. The environment variable "SUBPROC_TEST" is
# used to determine whether a test is skipped or not, such that if you want to
# run any OpenMP test directly this environment variable can be set. The
# subprocesses running the test classes set this environment variable as the new
# process starts which enables the tests within the process. The decorator
# @needs_subprocess is used to ensure the appropriate test skips are made.

#
# class TestOpenmpRunner(TestCase):
#    _numba_parallel_test_ = False
#
#    # Each test class can run for 30 minutes before time out.
#    _TIMEOUT = 1800
#
#    """This is the test runner for all the OpenMP tests, it runs them in
#    subprocesses as described above. The convention for the test method naming
#    is: `test_<TestClass>` where <TestClass> is the name of the test class in
#    this module.
#    """
#    def runner(self):
#        themod = self.__module__
#        test_clazz_name = self.id().split('.')[-1].split('_')[-1]
#        # don't specify a given test, it's an entire class that needs running
#        self.subprocess_test_runner(test_module=themod,
#                                    test_class=test_clazz_name,
#                                    timeout=self._TIMEOUT)
#
#    """
#    def test_TestOpenmpBasic(self):
#        self.runner()
#    """
#
#    def test_TestOpenmpRoutinesEnvVariables(self):
#        self.runner()
#
#    def test_TestOpenmpParallelForResults(self):
#        self.runner()
#
#    def test_TestOpenmpWorksharingSchedule(self):
#        self.runner()
#
#    def test_TestOpenmpParallelClauses(self):
#        self.runner()
#
#    def test_TestOpenmpDataClauses(self):
#        self.runner()
#
#    def test_TestOpenmpConstraints(self):
#        self.runner()
#
#    def test_TestOpenmpConcurrency(self):
#        self.runner()
#
#    def test_TestOpenmpTask(self):
#        self.runner()
#
#    def test_TestOpenmpTaskloop(self):
#        self.runner()
#
#    def test_TestOpenmpTarget(self):
#        self.runner()
#
#    def test_TestOpenmpPi(self):
#        self.runner()


x86_only = unittest.skipIf(
    platform.machine() not in ("i386", "x86_64"), "x86 only test"
)


def null_comparer(a, b):
    """
    Used with check_arq_equality to indicate that we do not care
    whether the value of the parameter at the end of the function
    has a particular value.
    """
    pass


@contextlib.contextmanager
def override_config(name, value):
    """
    Return a context manager that temporarily sets an openmp config variable
    *name* to *value*.  *name* must be the name of an existing variable
    in openmp.
    """
    old_value = getattr(numba.openmp, name)
    setattr(numba.openmp, name, value)
    try:
        yield
    finally:
        setattr(numba.openmp, name, old_value)


# @needs_subprocess
class TestOpenmpBase(TestCase):
    """
    Base class for testing OpenMP.
    Provides functions for compilation and three way comparison between
    python functions, njit'd functions and njit'd functions with
    OpenMP disabled.

    To set a default value or state for all the tests in a class, set
    a variable *var* inside the class where *var* is:

    - MAX_THREADS - Thread team size for parallel regions.
    - MAX_ACTIVE_LEVELS - Number of nested parallel regions capable of
                          running in parallel.
    """

    _numba_parallel_test_ = False

    skip_disabled = int(os.environ.get("OVERRIDE_TEST_SKIP", 0)) != 0
    run_target = int(os.environ.get("RUN_TARGET", 0)) != 0
    test_devices = os.environ.get("TEST_DEVICES", "")

    env_vars = {
        "OMP_NUM_THREADS": omp_get_num_procs(),
        "OMP_MAX_ACTIVE_LEVELS": 1,
        "OMP_DYNAMIC": True,
    }

    def __init__(self, *args):
        # flags for njit()
        self.cflags = Flags()
        self.cflags.enable_ssa = False
        self.cflags.nrt = True

        super(TestOpenmpBase, self).__init__(*args)

    def setUp(self):
        omp_set_num_threads(
            getattr(self, "MAX_THREADS", TestOpenmpBase.env_vars.get("OMP_NUM_THREADS"))
        )
        omp_set_max_active_levels(
            getattr(
                self,
                "MAX_ACTIVE_LEVELS",
                TestOpenmpBase.env_vars.get("OMP_MAX_ACTIVE_LEVELS"),
            )
        )
        self.beforeThreads = omp_get_max_threads()
        self.beforeLevels = omp_get_max_active_levels()

    def tearDown(self):
        omp_set_num_threads(self.beforeThreads)
        omp_set_max_active_levels(self.beforeLevels)

    def assert_outputs_equal(self, *outputs):
        assert len(outputs) > 1

        for op_num in range(len(outputs) - 1):
            op1, op2 = outputs[op_num], outputs[op_num + 1]
            if isinstance(op1, (bool, np.bool_)):
                assert isinstance(op2, (bool, np.bool_))
            elif not isinstance(op1, numbers.Number) or not isinstance(
                op2, numbers.Number
            ):
                self.assertEqual(type(op1), type(op2))

            if isinstance(op1, np.ndarray):
                np.testing.assert_almost_equal(op1, op2)
            elif isinstance(op1, (tuple, list)):
                assert len(op1) == len(op2)
                for i in range(len(op1)):
                    self.assert_outputs_equal(op1[i], op2[i])
            elif isinstance(op1, (bool, np.bool_, str, type(None))):
                assert op1 == op2
            elif isinstance(op1, numbers.Number):
                np.testing.assert_approx_equal(op1, op2)
            else:
                raise ValueError("Unsupported output type encountered")


class TestPipeline(object):
    def __init__(self, typingctx, targetctx, args, test_ir):
        self.state = compiler.StateDict()
        self.state.typingctx = typingctx
        self.state.targetctx = targetctx
        self.state.args = args
        self.state.func_ir = test_ir
        self.state.typemap = None
        self.state.return_type = None
        self.state.calltypes = None
        self.state.metadata = {}


#
# class TestOpenmpBasic(TestOpenmpBase):
#    """OpenMP smoke tests. These tests check the most basic
#    functionality"""
#
#    def __init__(self, *args):
#        TestOpenmpBase.__init__(self, *args)


class TestOpenmpRoutinesEnvVariables(TestOpenmpBase):
    MAX_THREADS = 5

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    """
    def test_func_get_wtime(self):
        @njit
        def test_impl(t):
            start = omp_get_wtime()
            time.sleep(t)
            return omp_get_wtime() - start
        t = 0.5
        np.testing.assert_approx_equal(test_impl(t), t, signifcant=2)
    """

    def test_func_get_max_threads(self):
        @njit
        def test_impl():
            omp_set_dynamic(0)
            o_nt = omp_get_max_threads()
            count = 0
            with openmp("parallel"):
                i_nt = omp_get_max_threads()
                with openmp("critical"):
                    count += 1
            return count, i_nt, o_nt

        nt = self.MAX_THREADS
        with override_env_config("OMP_NUM_THREADS", str(nt)):
            r = test_impl()
        assert r[0] == r[1] == r[2] == nt

    def test_func_get_num_threads(self):
        @njit
        def test_impl():
            omp_set_dynamic(0)
            o_nt = omp_get_num_threads()
            count = 0
            with openmp("parallel"):
                i_nt = omp_get_num_threads()
                with openmp("critical"):
                    count += 1
            return (count, i_nt), o_nt

        nt = self.MAX_THREADS
        with override_env_config("OMP_NUM_THREADS", str(nt)):
            r = test_impl()
        assert r[0][0] == r[0][1] == nt
        assert r[1] == 1

    def test_func_set_num_threads(self):
        @njit
        def test_impl(n1, n2):
            omp_set_dynamic(0)
            omp_set_num_threads(n1)
            count1 = 0
            count2 = 0
            with openmp("parallel"):
                with openmp("critical"):
                    count1 += 1
                omp_set_num_threads(n2)
            with openmp("parallel"):
                with openmp("critical"):
                    count2 += 1
            return count1, count2

        nt = 32
        with override_env_config("OMP_NUM_THREADS", str(4)):
            r = test_impl(nt, 20)
        assert r[0] == r[1] == nt

    def test_func_set_max_active_levels(self):
        @njit
        def test_impl(n1, n2, n3):
            omp_set_dynamic(0)
            omp_set_max_active_levels(2)
            omp_set_num_threads(n2)
            count1, count2, count3 = 0, 0, 0
            with openmp("parallel num_threads(n1)"):
                with openmp("single"):
                    with openmp("parallel"):
                        with openmp("single"):
                            omp_set_num_threads(n3)
                            with openmp("parallel"):
                                with openmp("critical"):
                                    count3 += 1
                        with openmp("critical"):
                            count2 += 1
                with openmp("critical"):
                    count1 += 1
            return count1, count2, count3

        n1, n2 = 3, 4
        r = test_impl(n1, n2, 5)
        assert r[0] == n1
        assert r[1] == n2
        assert r[2] == 1

    def test_func_get_ancestor_thread_num(self):
        @njit
        def test_impl():
            oa = omp_get_ancestor_thread_num(0)
            with openmp("parallel"):
                with openmp("single"):
                    m1 = omp_get_ancestor_thread_num(0)
                    f1 = omp_get_ancestor_thread_num(1)
                    s1 = omp_get_ancestor_thread_num(2)
                    tn1 = omp_get_thread_num()
                    with openmp("parallel"):
                        m2 = omp_get_ancestor_thread_num(0)
                        f2 = omp_get_ancestor_thread_num(1)
                        s2 = omp_get_ancestor_thread_num(2)
                        tn2 = omp_get_thread_num()
            return oa, (m1, f1, s1, tn1), (m2, f2, s2, tn2)

        oa, r1, r2 = test_impl()
        assert oa == r1[0] == r2[0] == 0
        assert r1[1] == r1[3] == r2[1]
        assert r1[2] == -1
        assert r2[2] == r2[3]

    def test_func_get_team_size(self):
        @njit
        def test_impl(n1, n2):
            omp_set_max_active_levels(2)
            oa = omp_get_team_size(0)
            with openmp("parallel num_threads(n1)"):
                with openmp("single"):
                    m1 = omp_get_team_size(0)
                    f1 = omp_get_team_size(1)
                    s1 = omp_get_team_size(2)
                    nt1 = omp_get_num_threads()
                    with openmp("parallel num_threads(n2)"):
                        with openmp("single"):
                            m2 = omp_get_team_size(0)
                            f2 = omp_get_team_size(1)
                            s2 = omp_get_team_size(2)
                            nt2 = omp_get_num_threads()
            return oa, (m1, f1, s1, nt1), (m2, f2, s2, nt2)

        n1, n2 = 6, 8
        oa, r1, r2 = test_impl(n1, n2)
        assert oa == r1[0] == r2[0] == 1
        assert r1[1] == r1[3] == r2[1] == n1
        assert r1[2] == -1
        assert r2[2] == r2[3] == n2

    def test_func_get_level(self):
        @njit
        def test_impl():
            oa = omp_get_level()
            with openmp("parallel if(0)"):
                f = omp_get_level()
                with openmp("parallel num_threads(1)"):
                    s = omp_get_level()
                    with openmp("parallel"):
                        t = omp_get_level()
            return oa, f, s, t

        for i, l in enumerate(test_impl()):
            assert i == l

    def test_func_get_active_level(self):
        @njit
        def test_impl():
            oa = omp_get_active_level()
            with openmp("parallel if(0)"):
                f = omp_get_active_level()
                with openmp("parallel num_threads(1)"):
                    s = omp_get_active_level()
                    with openmp("parallel"):
                        t = omp_get_active_level()
            return oa, f, s, t

        r = test_impl()
        for i in range(3):
            assert r[i] == 0
        assert r[3] == 1

    def test_func_in_parallel(self):
        @njit
        def test_impl():
            omp_set_dynamic(0)
            omp_set_max_active_levels(1)  # 1 because first region is inactive
            oa = omp_in_parallel()
            with openmp("parallel num_threads(1)"):
                ia = omp_in_parallel()
                with openmp("parallel"):
                    n1a = omp_in_parallel()
                    with openmp("single"):
                        with openmp("parallel"):
                            n2a = omp_in_parallel()
            with openmp("parallel if(0)"):
                ua = omp_in_parallel()
            return oa, ia, n1a, n2a, ua

        r = test_impl()
        assert r[0] == False
        assert r[1] == False
        assert r[2] == True
        assert r[3] == True
        assert r[4] == False

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_func_in_final(self):
        @njit
        def test_impl(N, c):
            a = np.arange(N)[::-1]
            fa = np.zeros(N)
            fia = np.zeros(N)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(len(a)):
                        e = a[i]
                        with openmp("task final(e >= c)"):
                            fa[i] = omp_in_final()
                            with openmp("task"):
                                fia[i] = omp_in_final()
            return fa, fia

        N, c = 25, 10
        r = test_impl(N, c)
        np.testing.assert_array_equal(r[0], np.concatenate(np.ones(N - c), np.zeros(c)))
        np.testing.assert_array_equal(r[0], r[1])


class TestOpenmpParallelForResults(TestOpenmpBase):
    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_parallel_for_set_elements(self):
        @njit
        def test_impl(v):
            with openmp("parallel for"):
                for i in range(len(v)):
                    v[i] = 1.0
            return v

        r = test_impl(np.zeros(100))
        np.testing.assert_array_equal(r, np.ones(100))

    def test_parallel_nested_for_set_elements(self):
        def test_impl(v):
            with openmp("parallel"):
                with openmp("for"):
                    for i in range(len(v)):
                        v[i] = 1.0
            return v

        r = test_impl(np.zeros(100))
        np.testing.assert_array_equal(r, np.ones(100))

    def test_parallel_for_const_var_omp_statement(self):
        def test_impl(v):
            ovar = "parallel for"
            with openmp(ovar):
                for i in range(len(v)):
                    v[i] = 1.0
            return v

        r = test_impl(np.zeros(100))
        np.testing.assert_array_equal(r, np.ones(100))

    def test_parallel_for_string_conditional(self):
        def test_impl(S):
            capitalLetters = 0
            with openmp("parallel for reduction(+:capitalLetters)"):
                for i in range(len(S)):
                    if S[i].isupper():
                        capitalLetters += 1
            return capitalLetters

        r = test_impl("OpenMPstrTEST")
        np.testing.assert_equal(r, 7)

    def test_parallel_for_tuple(self):
        def test_impl(t):
            len_total = 0
            with openmp("parallel for reduction(+:len_total)"):
                for i in range(len(t)):
                    len_total += len(t[i])
            return len_total

        r = test_impl(("32", "4", "test", "567", "re", ""))
        np.testing.assert_equal(r, 12)

    def test_parallel_for_range_step_2(self):
        @njit
        def test_impl(N):
            a = np.zeros(N, dtype=np.int32)
            with openmp("parallel for"):
                for i in range(0, len(a), 2):
                    a[i] = i + 1

            return a

        r = test_impl(12)
        np.testing.assert_array_equal(
            r, np.array([1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0], dtype=np.int32)
        )

    def test_parallel_for_range_step_arg(self):
        def test_impl(N, step):
            a = np.zeros(N, dtype=np.int32)
            with openmp("parallel for"):
                for i in range(0, len(a), step):
                    a[i] = i + 1

            return a

        r = test_impl(12, 2)
        np.testing.assert_array_equal(
            r, np.array([1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0], dtype=np.int32)
        )

    def test_parallel_for_incremented_step(self):
        @njit
        def test_impl(v, n):
            for i in range(n):
                with openmp("parallel for"):
                    for j in range(0, len(v), i + 1):
                        v[j] = i + 1
            return v

        r = test_impl(np.zeros(10), 3)
        np.testing.assert_array_equal(
            r, np.array([3.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 1.0, 2.0, 3.0])
        )

    def test_parallel_for_range_backward_step(self):
        @njit
        def test_impl(N):
            a = np.zeros(N, dtype=np.int32)
            with openmp("parallel for"):
                for i in range(N - 1, -1, -1):
                    a[i] = i + 1

            return a

        r = test_impl(12)
        np.testing.assert_array_equal(r, np.arange(1, 13, dtype=np.int32))

    """
    def test_parallel_for_dictionary(self):
        def test_impl(N, c):
            l = {}
            with openmp("parallel for"):
                for i in range(N):
                    l[i] = i % c
            return l
        # check
    """

    def test_parallel_for_num_threads(self):
        @njit
        def test_impl(nt):
            a = np.zeros(nt)
            with openmp("parallel num_threads(nt)"):
                with openmp("for"):
                    for i in range(nt):
                        a[i] = i
            return a

        r = test_impl(15)
        np.testing.assert_array_equal(r, np.arange(15))

    def test_parallel_for_only_inside_var(self):
        @njit
        def test_impl(nt):
            a = np.zeros(nt)
            with openmp("parallel num_threads(nt) private(x)"):
                with openmp("for private(x)"):
                    for i in range(nt):
                        x = 0
                        # print("out:", i, x, i + x, nt)
                        a[i] = i + x
            return a

        nt = 12
        np.testing.assert_array_equal(test_impl(nt), np.arange(nt))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_parallel_for_ordered(self):
        @njit
        def test_impl(N, c):
            a = np.zeros(N)
            b = np.zeros(N)
            with openmp("parallel for ordered"):
                for i in range(1, N):
                    b[i] = b[i - 1] + c
                    with openmp("ordered"):
                        a[i] = a[i - 1] + c
            return a

        N, c = 30, 4
        r = test_impl(N, c)
        rc = np.arange(0, N * c, c)
        np.testing.assert_array_equal(r[0], rc)
        assert not np.array_equal(r[1], rc)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_parallel_for_collapse(self):
        @njit
        def test_impl(n1, n2, n3):
            ia = np.zeros(n1)
            ja = np.zeros((n1, n2))
            ka = np.zeros((n1, n2, n3))
            with openmp("parallel for collapse(2)"):
                for i in range(n1):
                    ia[i] = omp_get_thread_num()
                    for j in range(n2):
                        ja[i][j] = omp_get_thread_num()
                        for k in range(n3):
                            ka[i][j][k] = omp_get_thread_num()
            return ia, ja, ka

        ia, ja, ka = test_impl(5, 3, 2)
        print(ia)
        print(ja)
        for a1i in range(len(ja)):
            with self.assertRaises(AssertionError) as raises:
                np.testing.assert_equal(ia[a1i], ja[a1i])  # Scalar to array
        for a1i in range(len(ka)):
            for a2i in range(a1i):
                # Scalar to array
                np.testing.assert_equal(ja[a1i][a2i], ka[a1i][a2i])


class TestOpenmpWorksharingSchedule(TestOpenmpBase):
    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    # Giorgis pass doesn't support static with chunksize yet?
    # @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Abort - unimplemented")
    # TODO: check the schedule
    def test_avg_sched_const(self):
        @njit
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            with openmp("parallel for num_threads(nt) schedule(static, 4)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i - 1]) / 2.0

            return b

        r = test_impl(10, np.ones(10))
        np.testing.assert_array_equal(
            r, [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Abort - unimplemented")
    def test_avg_sched_var(self):
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            ss = 4
            with openmp("parallel for num_threads(nt) schedule(static, ss)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i - 1]) / 2.0

            return b

        r = test_impl(10, np.ones(10))
        # create check

    def test_static_distribution(self):
        @njit
        def test_impl(nt, c):
            a = np.empty(nt * c)
            with openmp("parallel for num_threads(nt) schedule(static)"):
                for i in range(nt * c):
                    a[i] = omp_get_thread_num()
            return a

        nt, c = 8, 3
        r = test_impl(nt, c)
        for tn in range(nt):
            indices = np.sort(np.where(r == tn)[0])
            si = indices[0]
            np.testing.assert_array_equal(indices, np.arange(si, si + c))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_static_chunk_distribution(self):
        @njit
        def test_impl(nt, c, cs):
            a = np.empty(nt * c)
            with openmp("parallel for num_threads(nt) schedule(static, cs)"):
                for i in range(nt * c):
                    a[i] = omp_get_thread_num()
            return a

        nt, c, cs = 8, 6, 3
        r = test_impl(nt, c, cs)
        for tn in range(nt):
            indices = np.sort(np.where(r == tn)[0])
            for i in range(c // cs):
                si = indices[i * cs]
                np.testing.assert_array_equal(
                    indices, np.arange(si, min(len(r), si + cs))
                )

    def test_static_consistency(self):
        @njit
        def test_impl(nt, c, cs):
            a = np.empty(nt * c)
            b = np.empty(nt * c)
            with openmp("parallel num_threads(8)"):
                with openmp("for schedule(static)"):
                    for i in range(nt * c):
                        a[i] = omp_get_thread_num()
                with openmp("for schedule(static)"):
                    for i in range(nt * c):
                        b[i] = omp_get_thread_num()
            return a, b

        r = test_impl(8, 7, 5)
        np.testing.assert_array_equal(r[0], r[1])

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_dynamic_distribution(self):
        @njit
        def test_impl(nt, c, cs):
            a = np.empty(nt * c)
            with openmp("parallel for num_threads(nt) schedule(dynamic)"):
                for i in range(nt * c):
                    a[i] = omp_get_thread_num()
            return a

        nt, c, cs = 10, 2, 1
        r = test_impl(nt, c, cs)
        a = np.zeros(nt)
        for tn in range(nt):
            indices = np.sort(np.where(r == tn)[0])
            if len(indices > 0):
                for i in range(c // cs):
                    si = indices[i * cs]
                    np.testing.assert_array_equal(
                        indices, np.arange(si, min(len(r), si + cs))
                    )
            else:
                a[tn] = 1
        assert np.any(a)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_guided_distribution(self):
        @njit
        def test_impl(nt, c, cs):
            a = np.empty(nt * c)
            with openmp("parallel for num_threads(nt) schedule(guided, cs)"):
                for i in range(nt * c):
                    a[i] = omp_get_thread_num()
            return a

        nt, c, cs = 8, 6, 3
        r = test_impl(nt, c, cs)
        chunksizes = []
        cur_tn = r[0]
        cur_chunk = 0
        for e in r:
            if e == cur_tn:
                cur_chunk += 1
            else:
                chunksizes.append(cur_chunk)
                cur_chunk = 1
        chunksizes.append(cur_chunk)
        ca = np.array(chunksizes)
        np.testing.assert_array_equal(ca, np.sort(ca)[::-1])
        assert ca[-2] >= cs


class TestOpenmpParallelClauses(TestOpenmpBase):
    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_num_threads_clause(self):
        @njit
        def test_impl(N, c1, c2):
            omp_set_dynamic(0)
            omp_set_max_active_levels(2)
            omp_set_num_threads(N + c1)
            d_count = 0
            n_count = 0
            nc_count = 0
            a_count = 0
            with openmp("parallel num_threads(N) shared(c2)"):
                with openmp("critical"):
                    d_count += 1
                with openmp("parallel"):
                    with openmp("critical"):
                        n_count += 1
                with openmp("single"):
                    with openmp("parallel num_threads(6)"):
                        with openmp("critical"):
                            nc_count += 1
            with openmp("parallel"):
                with openmp("critical"):
                    a_count += 1
            return d_count, a_count, n_count, nc_count

        a, b, c = 13, 3, 6
        r = test_impl(a, b, c)
        assert r[0] == a
        assert r[1] == a + b
        assert r[2] == a * (a + b)
        assert r[3] == c

    def test_if_clause(self):
        @njit
        def test_impl(s):
            rp = 2  # Should also work with anything non-zero
            drp = 0
            ar = np.zeros(s, dtype=np.int32)
            adr = np.zeros(s, dtype=np.int32)
            par = np.full(s, 2, dtype=np.int32)
            padr = np.full(s, 2, dtype=np.int32)

            omp_set_num_threads(s)
            omp_set_dynamic(0)
            with openmp("parallel for if(rp)"):
                for i in range(s):
                    ar[omp_get_thread_num()] = 1
                    par[i] = omp_in_parallel()
            with openmp("parallel for if(drp)"):
                for i in range(s):
                    adr[omp_get_thread_num()] = 1
                    padr[i] = omp_in_parallel()
            return ar, adr, par, padr

        size = 20
        r = test_impl(size)
        np.testing.assert_array_equal(r[0], np.ones(size))
        rc = np.zeros(size)
        rc[0] = 1
        np.testing.assert_array_equal(r[1], rc)
        np.testing.assert_array_equal(r[2], np.ones(size))
        np.testing.assert_array_equal(r[3], np.zeros(size))

    def test_avg_arr_prev_two_elements_base(self):
        @njit
        def test_impl(n, a):
            b = np.zeros(n)
            omp_set_num_threads(5)

            with openmp("parallel for"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i - 1]) / 2.0
            return b

        r = test_impl(10, np.ones(10))
        np.testing.assert_array_equal(
            r, [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

    def test_avg_num_threads_clause(self):
        @njit
        def test_impl(n, a):
            b = np.zeros(n)
            with openmp("parallel for num_threads(5)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i - 1]) / 2.0

            return b

        r = test_impl(10, np.ones(10))
        np.testing.assert_array_equal(
            r, [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

    def test_avg_num_threads_clause_var(self):
        @njit
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            with openmp("parallel for num_threads(nt)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i - 1]) / 2.0

            return b

        r = test_impl(10, np.ones(10))
        np.testing.assert_array_equal(
            r, [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

    def test_avg_if_const(self):
        @njit
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            with openmp("parallel for if(1) num_threads(nt) schedule(static, 4)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i - 1]) / 2.0

            return b

        r = test_impl(10, np.ones(10))
        np.testing.assert_array_equal(
            r, [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Abort - unimplemented")
    def test_avg_if_var(self):
        @njit
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            ss = 4
            do_if = 1
            with openmp("parallel for if(do_if) num_threads(nt) schedule(static, ss)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i - 1]) / 2.0

            return b

        r = test_impl(10, np.ones(10))
        np.testing.assert_array_equal(
            r, [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

    def test_teams(self):
        @njit
        def test_impl():
            a = 1
            with openmp("teams"):
                with openmp("parallel"):
                    a = 123
            return a

        r = test_impl()
        np.testing.assert_equal(r, 123)


class TestReductions(TestOpenmpBase):
    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_parallel_reduction_add_int(self):
        @njit
        def test_impl():
            redux = 0
            nthreads = 0
            with openmp("parallel reduction(+:redux)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = 1
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, nthreads)

    def test_parallel_reduction_sub_int(self):
        @njit
        def test_impl():
            redux = 0
            nthreads = 0
            with openmp("parallel reduction(-:redux)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = 1
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, nthreads)

    def test_parallel_reduction_mul_int(self):
        @njit
        def test_impl():
            redux = 1
            nthreads = 0
            with openmp("parallel reduction(*:redux) num_threads(8)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = 2
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, 2**nthreads)

    def test_parallel_reduction_add_fp64(self):
        @njit
        def test_impl():
            redux = np.float64(0.0)
            nthreads = np.float64(0.0)
            with openmp("parallel reduction(+:redux)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = np.float64(1.0)
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, 1.0 * nthreads)

    def test_parallel_reduction_sub_fp64(self):
        @njit
        def test_impl():
            redux = np.float64(0.0)
            nthreads = np.float64(0.0)
            with openmp("parallel reduction(-:redux)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = np.float64(1.0)
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, 1.0 * nthreads)

    def test_parallel_reduction_mul_fp64(self):
        @njit
        def test_impl():
            redux = np.float64(1.0)
            nthreads = np.float64(0.0)
            with openmp("parallel reduction(*:redux) num_threads(8)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = np.float64(2.0)
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, 2.0**nthreads)

    def test_parallel_reduction_add_fp32(self):
        @njit
        def test_impl():
            redux = np.float32(0.0)
            nthreads = np.float32(0.0)
            with openmp("parallel reduction(+:redux)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = np.float32(1.0)
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, 1.0 * nthreads)

    def test_parallel_reduction_sub_fp32(self):
        @njit
        def test_impl():
            redux = np.float32(0.0)
            nthreads = np.float32(0.0)
            with openmp("parallel reduction(-:redux)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = np.float32(1.0)
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, 1.0 * nthreads)

    def test_parallel_reduction_mul_fp32(self):
        @njit
        def test_impl():
            redux = np.float32(1.0)
            nthreads = np.float32(0.0)
            with openmp("parallel reduction(*:redux) num_threads(8)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = np.float32(2.0)
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, 2.0**nthreads)

    def test_parallel_for_reduction_add_int(self):
        @njit
        def test_impl():
            redux = 0
            with openmp("parallel for reduction(+:redux)"):
                for i in range(10):
                    redux += 1
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10)

    def test_parallel_for_reduction_sub_int(self):
        @njit
        def test_impl():
            redux = 0
            with openmp("parallel for reduction(-:redux)"):
                for i in range(10):
                    redux += 1
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10)

    def test_parallel_for_reduction_mul_int(self):
        @njit
        def test_impl():
            redux = 1
            with openmp("parallel for reduction(*:redux)"):
                for i in range(10):
                    redux *= 2
            return redux

        redux = test_impl()
        self.assertEqual(redux, 2**10)

    def test_parallel_for_reduction_add_fp64(self):
        @njit
        def test_impl():
            redux = np.float64(0.0)
            with openmp("parallel for reduction(+:redux)"):
                for i in range(10):
                    redux += np.float64(1.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10.0)

    def test_parallel_for_reduction_sub_fp64(self):
        @njit
        def test_impl():
            redux = np.float64(0.0)
            with openmp("parallel for reduction(-:redux)"):
                for i in range(10):
                    redux += np.float64(1.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10.0)

    def test_parallel_for_reduction_mul_fp64(self):
        @njit
        def test_impl():
            redux = np.float64(1.0)
            with openmp("parallel for reduction(*:redux)"):
                for i in range(10):
                    redux *= np.float64(2.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 2.0**10)

    def test_parallel_for_reduction_add_fp32(self):
        @njit
        def test_impl():
            redux = np.float32(0.0)
            with openmp("parallel for reduction(+:redux)"):
                for i in range(10):
                    redux += np.float32(1.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10.0)

    def test_parallel_for_reduction_sub_fp32(self):
        @njit
        def test_impl():
            redux = np.float32(0.0)
            with openmp("parallel for reduction(-:redux)"):
                for i in range(10):
                    redux += np.float32(1.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10.0)

    def test_parallel_for_reduction_mul_fp32(self):
        @njit
        def test_impl():
            redux = np.float32(1.0)
            with openmp("parallel for reduction(*:redux)"):
                for i in range(10):
                    redux *= np.float32(2.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 2.0**10)

    def test_parallel_nest_for_reduction_add_int(self):
        @njit
        def test_impl():
            redux = 0
            with openmp("parallel"):
                with openmp("for reduction(+:redux)"):
                    for i in range(10):
                        redux += 1
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10)

    def test_parallel_nest_for_reduction_sub_int(self):
        @njit
        def test_impl():
            redux = 0
            with openmp("parallel"):
                with openmp("for reduction(-:redux)"):
                    for i in range(10):
                        redux += 1
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10)

    def test_parallel_nest_for_reduction_mul_int(self):
        @njit
        def test_impl():
            redux = 1
            with openmp("parallel"):
                with openmp("for reduction(*:redux)"):
                    for i in range(10):
                        redux *= 2
            return redux

        redux = test_impl()
        self.assertEqual(redux, 2**10)

    def test_parallel_nest_for_reduction_add_fp64(self):
        @njit
        def test_impl():
            redux = np.float64(0.0)
            with openmp("parallel"):
                with openmp("for reduction(+:redux)"):
                    for i in range(10):
                        redux += np.float64(1.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10.0)

    def test_parallel_nest_for_reduction_sub_fp64(self):
        @njit
        def test_impl():
            redux = np.float64(0.0)
            with openmp("parallel"):
                with openmp("for reduction(-:redux)"):
                    for i in range(10):
                        redux += np.float64(1.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10.0)

    def test_parallel_nest_for_reduction_mul_fp64(self):
        @njit
        def test_impl():
            redux = np.float64(1.0)
            with openmp("parallel"):
                with openmp("for reduction(*:redux)"):
                    for i in range(10):
                        redux *= np.float64(2.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 2.0**10)

    def test_parallel_nest_for_reduction_add_fp32(self):
        @njit
        def test_impl():
            redux = np.float32(0.0)
            with openmp("parallel"):
                with openmp("for reduction(+:redux)"):
                    for i in range(10):
                        redux += np.float32(1.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10.0)

    def test_parallel_nest_for_reduction_sub_fp32(self):
        @njit
        def test_impl():
            redux = np.float32(0.0)
            with openmp("parallel"):
                with openmp("for reduction(-:redux)"):
                    for i in range(10):
                        redux += np.float32(1.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10.0)

    def test_parallel_nest_for_reduction_mul_fp32(self):
        @njit
        def test_impl():
            redux = np.float32(1.0)
            with openmp("parallel"):
                with openmp("for reduction(*:redux)"):
                    for i in range(10):
                        redux *= np.float32(2.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 2.0**10)

    def test_parallel_reduction_add_int_10(self):
        @njit
        def test_impl():
            redux = 10
            nthreads = 0
            with openmp("parallel reduction(+:redux)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = 1
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, nthreads + 10)

    def test_parallel_reduction_add_fp32_10(self):
        @njit
        def test_impl():
            redux = np.float32(10.0)
            nthreads = np.float32(0.0)
            with openmp("parallel reduction(+:redux)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = np.float32(1.0)
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, 1.0 * nthreads + 10.0)

    def test_parallel_reduction_add_fp64_10(self):
        @njit
        def test_impl():
            redux = np.float64(10.0)
            nthreads = np.float64(0.0)
            with openmp("parallel reduction(+:redux)"):
                thread_id = omp_get_thread_num()
                if thread_id == 0:
                    nthreads = omp_get_num_threads()
                redux = np.float64(1.0)
            return redux, nthreads

        redux, nthreads = test_impl()
        self.assertGreater(nthreads, 1)
        self.assertEqual(redux, 1.0 * nthreads + 10.0)

    def test_parallel_for_reduction_add_int_10(self):
        @njit
        def test_impl():
            redux = 10
            with openmp("parallel for reduction(+:redux)"):
                for i in range(10):
                    redux += 1
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10 + 10)

    def test_parallel_for_reduction_add_fp32_10(self):
        @njit
        def test_impl():
            redux = np.float32(10.0)
            with openmp("parallel for reduction(+:redux)"):
                for i in range(10):
                    redux += np.float32(1.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10.0 + 10.0)

    def test_parallel_for_reduction_add_fp64_10(self):
        @njit
        def test_impl():
            redux = np.float64(10.0)
            with openmp("parallel for reduction(+:redux)"):
                for i in range(10):
                    redux += np.float64(1.0)
            return redux

        redux = test_impl()
        self.assertEqual(redux, 10.0 + 10.0)


class TestOpenmpDataClauses(TestOpenmpBase):
    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_default_none(self):
        @njit
        def test_impl(N):
            a = np.zeros(N, dtype=np.int32)
            x = 7
            with openmp("parallel for default(none)"):
                for i in range(N):
                    y = i + x
                    a[i] = y
                    z = i

            return a, z

        with self.assertRaises(UnspecifiedVarInDefaultNone) as raises:
            test_impl(100)
        self.assertIn("Variables with no data env clause", str(raises.exception))

    def test_data_sharing_default(self):
        @njit
        def test_impl(N, M):
            x = np.zeros(N)
            y = np.zeros(N)
            z = 3.14
            i = 7
            with openmp("parallel private(i)"):
                yn = M + 1
                zs = z
                with openmp("for"):
                    for i in range(N):
                        y[i] = yn + 2 * (i + 1)
                with openmp("for"):
                    for i in range(N):
                        x[i] = y[i] - i
                        with openmp("critical"):
                            z += 3
            return x, y, zs, z, i

        N, M = 10, 5
        r = test_impl(N, M)
        np.testing.assert_array_equal(r[0], np.arange(M + 3, M + N + 3))
        np.testing.assert_array_equal(r[1], np.arange(M + 3, M + 2 * N + 2, 2))
        assert r[2] == 3.14
        assert r[3] == 3.14 + 3 * N
        assert r[4] == 7

    def test_variables(self):
        @njit
        def test_impl():
            x = 5
            y = 3
            zfp = 2
            zsh = 7
            nerr = 0
            nsing = 0
            NTHREADS = 4
            numthrds = 0
            omp_set_num_threads(NTHREADS)
            vals = np.zeros(NTHREADS)
            valsfp = np.zeros(NTHREADS)

            with openmp("""parallel private(x) shared(zsh)
                        firstprivate(zfp) private(ID)"""):
                ID = omp_get_thread_num()
                with openmp("single"):
                    nsing = nsing + 1
                    numthrds = omp_get_num_threads()
                    if y != 3:
                        nerr = nerr + 1
                        print(
                            "Shared Default status failure y = ",
                            y,
                            " It should equal 3",
                        )

                # verify each thread sees the same variable vsh
                with openmp("critical"):
                    zsh = zsh + ID

                # test first private
                zfp = zfp + ID
                valsfp[ID] = zfp

                # setup test to see if each thread got its own x value
                x = ID
                vals[ID] = x

            # Shared clause test: assumes zsh starts at 7 and we add up IDs from 4 threads
            if zsh != 13:
                print("Shared clause or critical failed", zsh)
                nerr = nerr + 1

            # Single Test: How many threads updated nsing?
            if nsing != 1:
                print(" Single test failed", nsing)
                nerr = nerr + 1

            # Private clause test: did each thread get its own x variable?
            for i in range(numthrds):
                if int(vals[i]) != i:
                    print("Private clause failed", numthrds, i, vals[i])
                    nerr = nerr + 1

            # First private clause test: each thread should get 2 + ID for up to 4 threads
            for i in range(numthrds):
                if int(valsfp[i]) != 2 + i:
                    print("Firstprivate clause failed", numthrds, i, valsfp[i])
                    nerr = nerr + 1

            # Test number of threads
            if numthrds > NTHREADS:
                print("Number of threads error: too many threads", numthrds, NTHREADS)
                nerr = nerr + 1

            if nerr > 0:
                print(
                    nerr,
                    """ errors when testing parallel, private, shared,
                            firstprivate, critical  and single""",
                )

            return nerr

        assert test_impl() == 0

    def test_privates(self):
        @njit
        def test_impl(N):
            a = np.zeros(N, dtype=np.int32)
            x = 7
            with openmp("""parallel for firstprivate(x) private(y)
                         lastprivate(zzzz) private(private_index) shared(a)
                          firstprivate(N) default(none)"""):
                for private_index in range(N):
                    y = private_index + x
                    a[private_index] = y
                    zzzz = private_index

            return a, zzzz

        r, z = test_impl(10)
        np.testing.assert_array_equal(r, np.arange(7, 17))
        np.testing.assert_equal(z, 9)

    def test_private_retain_value(self):
        @njit
        def test_impl():
            x = 5
            with openmp("parallel private(x)"):
                x = 13
            return x

        assert test_impl() == 5

    def test_private_retain_value_param(self):
        @njit
        def test_impl(x):
            with openmp("parallel private(x)"):
                x = 13
            return x

        assert test_impl(5) == 5

    def test_private_retain_value_for(self):
        @njit
        def test_impl():
            x = 5
            with openmp("parallel private(x)"):
                with openmp("for"):
                    for i in range(10):
                        x = i
            return x

        assert test_impl() == 5

    def test_private_retain_value_for_param(self):
        @njit
        def test_impl(x):
            with openmp("parallel private(x)"):
                with openmp("for"):
                    for i in range(10):
                        x = i
            return x

        assert test_impl(5) == 5

    def test_private_retain_value_combined_for(self):
        @njit
        def test_impl():
            x = 5
            with openmp("parallel for private(x)"):
                for i in range(10):
                    x = i
            return x

        assert test_impl() == 5

    def test_private_retain_value_combined_for_param(self):
        @njit
        def test_impl(x):
            with openmp("parallel for private(x)"):
                for i in range(10):
                    x = i
            return x

        assert test_impl(5) == 5

    def test_private_retain_two_values(self):
        @njit
        def test_impl():
            x = 5
            y = 7
            with openmp("parallel private(x,y)"):
                x = 13
                y = 40
            return x, y

        assert test_impl() == (5, 7)

    def test_private_retain_array(self):
        @njit
        def test_impl(N, x):
            a = np.ones(N)
            with openmp("parallel private(a)"):
                with openmp("single"):
                    sa = a
                a = np.zeros(N)
                with openmp("for"):
                    for i in range(N):
                        a[i] = x
            return a, sa

        r = test_impl(10, 3)
        np.testing.assert_array_equal(r[0], np.ones(r[0].shape))
        with self.assertRaises(AssertionError) as raises:
            np.testing.assert_array_equal(r[1], np.ones(r[0].shape))

    def test_private_divide_work(self):
        @njit
        def test_impl(v, npoints):
            omp_set_num_threads(3)

            with openmp("""parallel default(shared)
                        private(iam,nt,ipoints,istart)"""):
                iam = omp_get_thread_num()
                nt = omp_get_num_threads()
                ipoints = npoints // nt
                istart = iam * ipoints
                if iam == nt - 1:
                    ipoints = npoints - istart
                for i in range(ipoints):
                    v[istart + i] = 123.456
            return v

        r = test_impl(np.zeros(12), 12)
        np.testing.assert_array_equal(r, np.full(12, 123.456))

    def test_firstprivate(self):
        @njit
        def test_impl(x, y):
            with openmp("parallel firstprivate(x)"):
                xs = x
                x = y
            return xs, x

        x, y = 5, 3
        self.assert_outputs_equal(test_impl(x, y), (x, x))

    def test_lastprivate_for(self):
        @njit
        def test_impl(N):
            a = np.zeros(N)
            si = 0
            with openmp("parallel for lastprivate(si)"):
                for i in range(N):
                    si = i + 1
                    a[i] = si
            return si, a

        N = 10
        r = test_impl(N)
        assert r[0] == N
        np.testing.assert_array_equal(r[1], np.arange(1, N + 1))

    def test_lastprivate_non_one_step(self):
        @njit
        def test_impl(n1, n2, s):
            a = np.zeros(math.ceil((n2 - n1) / s))
            rl = np.arange(n1, n2, s)
            with openmp("parallel for lastprivate(si)"):
                for i in range(len(rl)):
                    si = rl[i] + 1
                    a[i] = si
            return si, a

        n1, n2, s = 4, 26, 3
        r = test_impl(n1, n2, s)
        ra = np.arange(n1, n2, s) + 1
        assert r[0] == ra[-1]
        np.testing.assert_array_equal(r[1], ra)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_lastprivate_sections(self):
        @njit
        def test_impl(N2, si):
            a = np.zeros(N2)
            with openmp("parallel shared(sis1)"):
                with openmp("sections lastprivate(si)"):
                    sis1 = si
                    # N1 = number of sections
                    with openmp("section"):
                        si = 0
                    with openmp("section"):
                        si = 1
                    with openmp("section"):
                        si = 2
                sis2 = si
                with openmp("sections lastprivate(si)"):
                    # N2 = number of sections
                    with openmp("section"):
                        i = 0
                        si = N2 - i
                        a[i] = si
                    with openmp("section"):
                        i = 1
                        si = N2 - i
                        a[i] = si
                    with openmp("section"):
                        i = 2
                        si = N2 - i
                        a[i] = si
                    with openmp("section"):
                        i = 3
                        si = N2 - i
                        a[i] = si
            return si, sis1, sis2, a

        N1, N2, d = 3, 4, 5
        r = test_impl(N2, d)
        assert r[0] == 1
        assert r[1] != d
        assert r[2] == N1 - 1
        np.testing.assert_array_equal(r[3], np.arange(N2, 0, -1))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_lastprivate_conditional(self):
        @njit
        def test_impl(N, c1, c2):
            a = np.arange(0, N * 2, c2)
            num = 0
            with openmp("parallel"):
                with openmp("for lastprivate(conditional: num)"):
                    for i in range(N):
                        if i < c1:
                            num = a[i] + c2
            return num

        c1, c2 = 11, 3
        assert test_impl(15, c1, c2) == c1 * c2

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_threadprivate(self):
        @njit
        def test_impl(N, c):
            omp_set_num_threads(N)
            a = np.zeros(N)
            ra = np.zeros(N)
            val = 0
            with openmp("threadprivate(val)"):
                pass
            with openmp("parallel private(tn, sn)"):
                tn = omp_get_thread_num()
                sn = c + tn
                val = sn
                a[tn] = sn
            with openmp("parallel private(tn)"):
                tn = omp_get_thread_num()
                ra[tn] = 1 if val == a[tn] else 0
            return ra

        nt = 8
        np.testing.assert_array_equal(test_impl(nt, 5), np.ones(nt))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_copyin(self):
        @njit
        def test_impl(nt, n1, n2, n3):
            xsa1 = np.zeros(nt)
            xsa2 = np.zeros(nt)
            x = n1
            with openmp("threadprivate(x)"):
                pass
            x = n2
            with openmp("parallel num_threads(nt) copyin(x) private(tn)"):
                tn = omp_get_thread_num()
                xsa1[tn] = x
                if tn == 0:
                    x = n3
            with openmp("parallel copyin(x)"):
                xsa2[omp_get_thread_num()] = x
            return xsa1, xsa2

        nt, n2, n3 = 10, 12.5, 7.1
        r = test_impl(nt, 4.3, n2, n3)
        np.testing.assert_array_equal(r[0], np.full(nt, n2))
        np.testing.assert_array_equal(r[1], np.full(nt, n3))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_copyin_nested(self):
        def test_impl(nt1, nt2, mt, n1, n2, n3):
            omp_set_nested(1)
            omp_set_dynamic(0)
            xsa1 = np.zeros(nt1)
            xsa2 = np.zeros(nt2)
            x = n1
            with openmp("threadprivate(x)"):
                pass
            x = n2
            with openmp("parallel num_threads(nt1) copyin(x) private(tn)"):
                tn = omp_get_thread_num()
                xsa1[tn] = x
                if tn == mt:
                    x = n3
                    with openmp("parallel num_threads(nt2) copyin(x)"):
                        xsa2[omp_get_thread_num()] = x
            return xsa1, xsa2

        nt1, nt2, n2, n3 = 10, 4, 12.5, 7.1
        r = test_impl(nt1, nt2, 2, 4.3, n2, n3)
        np.testing.assert_array_equal(r[0], np.full(nt1, n2))
        np.testing.assert_array_equal(r[1], np.full(nt2, n3))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_copyprivate(self):
        @njit
        def test_impl(nt, n1, n2, n3):
            x = n1
            a = np.zeros(nt)
            xsa = np.zeros(nt)
            ar = np.zeros(nt)
            omp_set_num_threads(nt)
            with openmp("parallel firstprivate(x, a) private(tn)"):
                with openmp("single copyprivate(x, a)"):
                    x = n2
                    a = np.full(nt, n3)
                tn = omp_get_thread_num()
                xsa[tn] = x
                ar[tn] = a[tn]
            return xsa, a, ar

        nt, n2, n3 = 16, 12, 3
        r = test_impl(nt, 5, n2, n3)
        np.testing.assert_array_equal(r[0], np.full(nt, n2))
        self.assert_outputs_equal(r[1], r[2], np.full(nt, n3))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_linear_clause(self):
        @njit
        def test_impl(N):
            a = np.arange(N) + 1
            b = np.zeros(N // 2)

            linearj = 0
            with openmp("parallel for linear(linearj:1)"):
                for i in range(0, N, 2):
                    b[linearj] = a[i] * 2

            return b, linearj

        N = 50
        r = test_impl(N)
        np.testing.assert_array_equal(r[0], np.arange(2, N * 2 - 1, 4))
        assert r[1] == N // 2 - 1


class TestOpenmpConstraints(TestOpenmpBase):
    """Tests designed to confirm that errors occur when expected, or
    to see how OpenMP behaves in various circumstances"""

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_parallel_for_no_for_loop(self):
        @njit
        def test_impl():
            with openmp("parallel for"):
                pass

        with self.assertRaises(ParallelForWrongLoopCount) as raises:
            test_impl()
        self.assertIn(
            "OpenMP parallel for regions must contain exactly one",
            str(raises.exception),
        )

    def test_parallel_for_multiple_for_loops(self):
        @njit
        def test_impl():
            a = np.zeros(4)
            with openmp("parallel for"):
                for i in range(2):
                    a[i] = 1
                for i in range(2, 4):
                    a[i] = 1

        with self.assertRaises(ParallelForWrongLoopCount) as raises:
            test_impl()
        self.assertIn(
            "OpenMP parallel for regions must contain exactly one",
            str(raises.exception),
        )

    def test_statement_before_parallel_for(self):
        @njit
        def test_impl():
            a = np.zeros(4)
            with openmp("parallel for"):
                print("Fail")
                for i in range(4):
                    a[i] = i
            return a

        with self.assertRaises(ParallelForExtraCode) as raises:
            test_impl()
        self.assertIn("Extra code near line", str(raises.exception))

    def test_statement_after_parallel_for(self):
        @njit
        def test_impl():
            a = np.zeros(4)
            with openmp("parallel for"):
                for i in range(4):
                    a[i] = i
                print("Fail")
            return a

        with self.assertRaises(ParallelForExtraCode) as raises:
            a = test_impl()
            print("a", a)
        self.assertIn("Extra code near line", str(raises.exception))

    def test_nonstring_var_omp_statement(self):
        @njit
        def test_impl(v):
            ovar = 7
            with openmp(ovar):
                for i in range(len(v)):
                    v[i] = 1.0
            return v

        with self.assertRaises(NonStringOpenmpSpecification) as raises:
            test_impl(np.zeros(100))
        self.assertIn("Non-string OpenMP specification at line", str(raises.exception))

    def test_parallel_for_nonconst_var_omp_statement(self):
        @njit
        def test_impl(v):
            ovar = "parallel "
            ovar += "for"
            with openmp(ovar):
                for i in range(len(v)):
                    v[i] = 1.0
            return v

        with self.assertRaises(NonconstantOpenmpSpecification) as raises:
            test_impl(np.zeros(100))
        self.assertIn(
            "Non-constant OpenMP specification at line", str(raises.exception)
        )

    # def test_parallel_for_blocking_if(self):
    #    @njit
    #    def test_impl():
    #        n = 0
    #        with openmp("parallel"):
    #            half_threads = omp_get_num_threads()//2
    #            if omp_get_thread_num() < half_threads:
    #                with openmp("for reduction(+:n)"):
    #                    for _ in range(half_threads):
    #                        n += 1
    #        return n

    #    #with self.assertRaises(AssertionError) as raises:
    #     #   njit(test_impl)
    #    test_impl()
    #    #print(str(raises.exception))

    def test_parallel_for_delaying_condition(self):
        @njit
        def test_impl():
            n = 0
            with openmp("parallel private(lc)"):
                lc = 0
                while lc < omp_get_thread_num():
                    lc += 1
                with openmp("for reduction(+:n)"):
                    for _ in range(omp_get_num_threads()):
                        n += 1
            return n

        test_impl()

    def test_parallel_for_nowait(self):
        @njit
        def test_impl(nt):
            a = np.zeros(nt)
            with openmp("parallel for num_threads(nt) nowait"):
                for i in range(nt):
                    a[omp_get_thread_num] = i
            return a

        with self.assertRaises(Exception) as raises:
            test_impl(12)
        self.assertIn("No terminal matches", str(raises.exception))

    def test_parallel_double_num_threads(self):
        @njit
        def test_impl(nt1, nt2):
            count = 0
            with openmp("parallel num_threads(nt1) num_threads(nt2)"):
                with openmp("critical"):
                    count += 1
            print(count)
            return count

        with self.assertRaises(Exception) as raises:
            test_impl(5, 7)

    def test_conditional_barrier(self):
        @njit
        def test_impl(nt):
            hp = nt // 2
            a = np.zeros(hp)
            b = np.zeros(nt - hp)
            with openmp("parallel num_threads(nt) private(tn)"):
                tn = omp_get_thread_num()
                if tn < hp:
                    with openmp("barrier"):
                        pass
                    a[tn] = 1
                else:
                    with openmp("barrier"):
                        pass
                    b[tn - hp] = 1
            return a, b

        # The spec seems to say this should be an error but in practice maybe not?
        # with self.assertRaises(Exception) as raises:
        test_impl(12)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Hangs")
    def test_closely_nested_for_loops(self):
        @njit
        def test_impl(N):
            a = np.zeros((N, N))
            with openmp("parallel"):
                with openmp("for"):
                    for i in range(N):
                        with openmp("for"):
                            for j in range(N):
                                a[i][j] = 1
            return a

        with self.assertRaises(Exception) as raises:
            test_impl(4)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Hangs")
    def test_nested_critical(self):
        @njit
        def test_impl():
            num = 0
            with openmp("parallel"):
                with openmp("critical"):
                    num += 1
                    with openmp("critical"):
                        num -= 1
            return num

        with self.assertRaises(Exception) as raises:
            test_impl()


class TestOpenmpConcurrency(TestOpenmpBase):
    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_parallel_region(self):
        @njit
        def test_impl():
            a = 1
            with openmp("parallel"):
                a += 1

        test_impl()

    def test_single(self):
        @njit
        def test_impl(nt):
            omp_set_num_threads(nt)
            a = np.zeros(4, dtype=np.int64)
            with openmp("parallel"):
                with openmp("single"):
                    a[0] += 1
            return a

        np.testing.assert_array_equal(test_impl(4), np.array([1, 0, 0, 0]))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_master(self):
        @njit
        def test_impl(nt):
            omp_set_num_threads(nt)
            a = np.ones(4, dtype=np.int64)
            with openmp("parallel"):
                with openmp("master"):
                    a[0] += omp_get_thread_num()
            return a

        np.testing.assert_array_equal(test_impl(4), np.array([0, 1, 1, 1]))

    def test_critical_threads1(self):
        @njit
        def test_impl(N, iters):
            omp_set_num_threads(N)
            count = 0
            p = 0
            sum = 0
            with openmp("parallel"):
                with openmp("barrier"):
                    pass
                with openmp("for private(p, sum)"):
                    for _ in range(iters):
                        with openmp("critical"):
                            p = count
                            sum = 0
                            for i in range(10000):
                                if i % 2 == 0:
                                    sum += 1
                                else:
                                    sum -= 1
                            p += 1 + sum
                            count = p
            return count

        iters = 1000
        r = test_impl(2, iters)
        np.testing.assert_equal(r, iters)

    def test_critical_threads2(self):
        @njit
        def test_impl(N):
            omp_set_num_threads(N)
            ca = np.zeros(N)
            sum = 0
            with openmp("parallel private(sum) shared(c)"):
                c = N
                with openmp("barrier"):
                    pass
                with openmp("critical"):
                    ca[omp_get_thread_num()] = c - 1
                    # Sleep
                    sum = 0
                    for i in range(10000):
                        if i % 2 == 0:
                            sum += 1
                        else:
                            sum -= 1
                    c -= 1 + sum
            return np.sort(ca)

        nt = 16
        np.testing.assert_array_equal(test_impl(nt), np.arange(nt))

    def test_critical_result(self):
        @njit
        def test_impl(N):
            omp_set_num_threads(N)
            count = 0
            with openmp("parallel"):
                if omp_get_thread_num() < N // 2:
                    with openmp("critical"):
                        count += 1
                else:
                    with openmp("critical"):
                        count += 1
            return count

        nt = 16
        assert test_impl(nt) == nt

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_named_critical(self):
        @njit
        def test_impl(N):
            omp_set_num_threads(N)
            a = np.zeros((2, N))
            sa = np.zeros(N)
            with openmp("parallel private(a0c, sum, tn)"):
                tn = omp_get_thread_num()
                with openmp("barrier"):
                    pass
                with openmp("critical (a)"):
                    # Sleep
                    sum = 0
                    for j in range(1000):
                        if j % 2 == 0:
                            sum += 1
                        else:
                            sum -= 1
                    a[0][tn] = 1 + sum
                with openmp("critical (b)"):
                    a0c = np.copy(a[0])
                    # Sleep
                    sum = 0
                    for j in range(10000):
                        if j % 2 == 0:
                            sum += 1
                        else:
                            sum -= 1
                    a[1][tn] = 1 + sum
                    sa[tn] = 1 if a[0] != a0c else 0
            return a, sa

        nt = 16
        r = test_impl(nt)
        np.testing.assert_array_equal(r[0], np.ones((2, nt)))
        assert np.any(r[1])

    # Revisit - how to prove atomic works without a race condition?
    # def test_atomic_threads(self):
    #    def test_impl(N, iters):
    #        omp_set_num_threads(N)
    #        count = 0
    #        p = 0
    #        sum = 0
    #        with openmp("parallel"):
    #            with openmp("barrier"):
    #                pass
    #            with openmp("for private(p, sum)"):
    #                for _ in range(iters):
    #                    with openmp("atomic"):
    #                        p = count
    #                        sum = 0
    #                        for i in range(10000):
    #                            if i % 2 == 0:
    #                                sum += 1
    #                            else:
    #                                sum -= 1
    #                        p += 1 + sum
    #                        count = p
    #        return count
    #    iters = 1000
    #    r = test_impl(2, iters)
    #    create check

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_atomic(self):
        @njit
        def test_impl(nt, N, c):
            omp_set_num_threads(nt)
            a = np.zeros(N)
            with openmp("parallel for private(b, index)"):
                for i in range(nt):
                    b = 0
                    index = i % N
                    with openmp("atomic write"):
                        a[index] = nt % c
                    with openmp("barrier"):
                        pass
                    with openmp("atomic read"):
                        b = a[index - 1] + index
                    with openmp("barrier"):
                        pass
                    with openmp("atomic update"):
                        a[index] += b
            return a

        nt, N, c = 27, 8, 6
        rc = np.zeros(N)
        # ba = np.zeros(nt)
        # for i in range(nt):
        #    index = i % N
        #    rc[index] = nt % c
        # print("rc1:", rc)

        # for i in range(nt):
        #    index = i % N
        #    ba[i] = rc[index-1] + index

        # for i in range(nt):
        #    index = i % N
        #    rc[index] += ba[i]
        # print("rc2:", rc)

        for i in range(nt):
            index = i % N
            ts = nt // N
            ts += 1 if index < nt % N else 0
            rc[index] = nt % c + (nt % c + index) * ts

        np.testing.assert_array_equal(test_impl(nt, N, c), rc)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_atomic_capture(self):
        @njit
        def test_impl(nt, N, c):
            s = math.ceil(N // 2)
            a = np.zeros(s)
            sva = np.zeros(N)
            tns = np.zeros(N)
            with openmp("parallel for num_threads(nt) private(sv, index)"):
                for i in range(N):
                    index = i % s
                    tns[i] = omp_get_thread_num()
                    with openmp("atomic write"):
                        a[index] = index * c + 1
                    with openmp("barrier"):
                        pass
                    with openmp("atomic capture"):
                        sv = a[index - 1]
                        a[index - 1] += sv + (tns[i] % c + 1)
                    # sva[index] = sv
            return a, sva, tns

        nt, N, c = 16, 30, 7
        r1, r2, tns = test_impl(nt, N, c)
        size = math.ceil(N // 2)
        rc = np.arange(1, (size - 1) * c + 2, c)
        # np.testing.assert_array_equal(r2, np.roll(rc, 1))
        for i in range(N):
            index = i % size
            rc[index - 1] += rc[index - 1] + (tns[i] % c + 1)
        np.testing.assert_array_equal(r1, rc)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_parallel_sections(self):
        @njit
        def test_impl(nt):
            ta0 = np.zeros(nt)
            ta1 = np.zeros(nt)
            secpa = np.zeros(nt)

            with openmp("parallel sections num_threads(nt)"):
                with openmp("section"):
                    ta0[omp_get_thread_num()] += 1
                    secpa[0] = omp_in_parallel()
                with openmp("section"):
                    ta1[omp_get_thread_num()] += 1
                    secpa[1] = omp_in_parallel()
            print(ta0, ta1)
            return ta0, ta0, secpa

        NT = 2  # Must equal the number of section directives in the test
        r = test_impl(NT)
        assert np.sum(r[0]) == 1
        assert np.sum(r[1]) == 1
        assert np.sum(r[2]) == NT
        np.testing.assert_array_equal(r[0] + r[1], np.ones(NT))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Abort - needs fix")
    def test_barrier(self):
        @njit
        def test_impl(nt, iters, c):
            a = np.zeros(nt)
            ac = np.zeros((nt, nt))
            x = iters // c
            iters = x * c
            sum = 0
            with openmp("parallel num_threads(nt) private(tn, sum)"):
                tn = omp_get_thread_num()
                with openmp("critical"):
                    sum = 0
                    for i in range(iters):
                        if i % x == 0:
                            sum += 1
                    a[tn] = sum
                with openmp("barrier"):
                    pass
                for j in range(nt):
                    ac[tn][j] = a[j]
            return ac

        nt, c = 15, 12
        r = test_impl(nt, 10000, c)
        a = np.full(nt, c)
        for i in range(nt):
            np.testing.assert_array_equal(r[i], a)

    #    def test_for_nowait(self):
    #        @njit
    #        def test_impl(nt, n, c1, c2):
    #            a = np.zeros(n)
    #            b = np.zeros(n)
    #            ac = np.zeros((nt, n))
    #            sum = 0
    #            with openmp("parallel num_threads(nt) private(tn)"):
    #                tn = omp_get_thread_num()
    #                with openmp("for nowait schedule(static) private(sum)"):
    #                    for i in range(n):
    #                        # Sleep
    #                        sum = 0
    #                        for j in range(i * 1000):
    #                            if j % 2 == 0:
    #                                sum += 1
    #                            else:
    #                                sum -= 1
    #                        a[i] = i * c1 + sum
    #                for j in range(nt):
    #                    ac[tn][j] = a[j]
    #                with openmp("for schedule(static)"):
    #                    for i in range(n):
    #                        b[i] = a[i] + c2
    #            return b, ac
    #        nt, n, c1, c2 = 8, 30, 5, -7
    #        r = test_impl(nt, n, c1, c2)
    #        a = np.arange(n) * c1
    #        np.testing.assert_array_equal(r[0], a + c2)
    #        arc = [np.array_equal(r[1][i], a) for i in range(nt)]
    #        assert(not np.all(arc))
    #
    #    def test_nowait_result(self):
    #        def test_impl(n, m, a, b, y, z):
    #            omp_set_num_threads(5)
    #
    #            with openmp("parallel"):
    #                with openmp("for nowait"):
    #                    for i in range(1, n):
    #                        b[i] = (a[i] + a[i-1]) / 2.0
    #                with openmp("for nowait"):
    #                    for i in range(m):
    #                        y[i] = math.sqrt(z[i])
    #
    #            return b, y
    #        n, m = 10, 20
    #        r = test_impl(n, m, np.ones(n), np.zeros(n),
    #                    np.zeros(m), np.full(m, 13))
    # create check

    def test_nested_parallel_for(self):
        @njit
        def test_impl(nt):
            omp_set_num_threads(nt)
            omp_set_nested(1)
            omp_set_dynamic(0)
            a = np.zeros((nt, nt), dtype=np.int32)
            with openmp("parallel for"):
                for i in range(nt):
                    with openmp("parallel for"):
                        for j in range(nt):
                            a[i][j] = omp_get_thread_num()
            return a

        nt = 8
        r = test_impl(nt)
        for i in range(len(r)):
            np.testing.assert_array_equal(np.sort(r[i]), np.arange(nt))

    def test_nested_parallel_regions_1(self):
        @njit
        def test_impl(nt1, nt2):
            omp_set_dynamic(0)
            omp_set_max_active_levels(2)
            ca = np.zeros(nt1)
            omp_set_num_threads(nt1)
            with openmp("parallel private(tn)"):
                tn = omp_get_thread_num()
                with openmp("parallel num_threads(3)"):
                    with openmp("critical"):
                        ca[tn] += 1
                    with openmp("single"):
                        ats = omp_get_ancestor_thread_num(1) == tn
                        ts = omp_get_team_size(1)
            return ca, ats, ts

        nt1, nt2 = 6, 3
        r = test_impl(nt1, nt2)
        np.testing.assert_array_equal(r[0], np.full(nt1, nt2))
        assert r[1] == True
        assert r[2] == nt1

    def test_nested_parallel_regions_2(self):
        @njit
        def set_array(a):
            tn = omp_get_thread_num()
            a[tn][0] = omp_get_max_active_levels()
            a[tn][1] = omp_get_num_threads()
            a[tn][2] = omp_get_max_threads()
            a[tn][3] = omp_get_level()
            a[tn][4] = omp_get_team_size(1)
            a[tn][5] = omp_in_parallel()

        @njit
        def test_impl(mal, n1, n2, n3):
            omp_set_max_active_levels(mal)
            omp_set_dynamic(0)
            omp_set_num_threads(n1)
            a = np.zeros((n2, 6), dtype=np.int32)
            b = np.zeros((n1, 6), dtype=np.int32)
            with openmp("parallel"):
                omp_set_num_threads(n2)
                with openmp("single"):
                    with openmp("parallel"):
                        omp_set_num_threads(n3)
                        set_array(a)
                set_array(b)

            return a, b

        mal, n1, n2, n3 = 8, 2, 4, 5
        a, b = test_impl(mal, n1, n2, n3)
        for i in range(n2):
            np.testing.assert_array_equal(a[i], np.array([8, n2, n3, 2, n1, 1]))
        for i in range(n1):
            np.testing.assert_array_equal(b[i], np.array([8, n1, n2, 1, n1, 1]))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Abort / Segmentation Fault")
    def test_parallel_two_dimensional_array(self):
        @njit
        def test_impl(N):
            omp_set_dynamic(0)
            omp_set_num_threads(N)
            a = np.zeros((N, 2), dtype=np.int32)
            with openmp("parallel private(tn)"):
                tn = omp_get_thread_num()
                a[tn][0] = 1
                a[tn][1] = 2
            return a

        N = 5
        r = test_impl(N)
        for i in range(N):
            np.testing.assert_array_equal(r[i], np.array([1, 2]))


class TestOpenmpTask(TestOpenmpBase):
    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_task_basic(self):
        @njit
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task"):
                            a[i] = 1
            return a

        r = test_impl(15)
        np.testing.assert_array_equal(r, np.ones(15))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Sometimes segmentation fault")
    def test_task_thread_assignment(self):
        @njit
        def test_impl(ntsks):
            a = np.empty(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task"):
                            a[i] = omp_get_thread_num()
            return a

        with self.assertRaises(AssertionError) as raises:
            v = test_impl(15)
            np.testing.assert_equal(v[0], v)

    def test_task_data_sharing_default(self):
        @njit
        def test_impl(n1, n2):
            x = n1
            with openmp("parallel private(y)"):
                y = n1
                with openmp("single"):
                    with openmp("task"):
                        xa = x == n1
                        ya = y == n1
                        x, y = n2, n2
                    with openmp("taskwait"):
                        ysave = y
            return (x, ysave), (xa, ya)

        n1, n2 = 1, 2
        r = test_impl(n1, n2)
        self.assert_outputs_equal(r[1], (True, True))
        self.assert_outputs_equal(r[0], (n2, n1))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Segmentation fault")
    def test_task_single_implicit_barrier(self):
        @njit
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task private(sum)"):
                            # Sleep
                            sum = 0
                            for j in range(10000):
                                if j % 2 == 0:
                                    sum += 1
                                else:
                                    sum -= 1
                            a[i] = 1 + sum
                # with openmp("barrier"):
                #    pass
                sa = np.copy(a)
            return sa

        ntsks = 15
        r = test_impl(ntsks)
        np.testing.assert_array_equal(r, np.ones(ntsks))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Segmentation fault")
    def test_task_single_nowait(self):
        @njit
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single nowait"):
                    for i in range(ntsks):
                        with openmp("task private(sum)"):
                            sum = 0
                            for j in range(10000):
                                if j % 2 == 0:
                                    sum += 1
                                else:
                                    sum -= 1
                            a[i] = 1 + sum
                sa = np.copy(a)
            return sa

        with self.assertRaises(AssertionError) as raises:
            ntsks = 15
            r = test_impl(ntsks)
            np.testing.assert_array_equal(r, np.ones(ntsks))

    # Error with commented out code, other version never finished running
    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Error")
    def test_task_barrier(self):
        @njit
        def test_impl(nt):
            omp_set_num_threads(nt)
            a = np.zeros((nt + 1) * nt / 2)
            # a = np.zeros(10)
            with openmp("parallel"):
                with openmp("single"):
                    for tn in range(nt):
                        with openmp("task"):
                            for i in range(tn + 1):
                                with openmp("task"):
                                    a[i] = omp_get_thread_num() + 1
                    with openmp("barrier"):
                        ret = np.all(a)
            return ret

        assert test_impl(4)

    def test_taskwait(self):
        @njit
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel private(i)"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task private(sum) private(j)"):
                            sum = 0
                            for j in range(10000):
                                if j % 2 == 0:
                                    sum += 1
                                else:
                                    sum -= 1
                            a[i] = 1 + sum
                    with openmp("taskwait"):
                        ret = np.sum(a)
            return ret

        r = test_impl(15)
        np.testing.assert_equal(r, 15)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Sometimes segmentation fault")
    def test_taskwait_descendants(self):
        @njit
        def test_impl(ntsks, dtsks):
            a = np.zeros(ntsks)
            da = np.zeros((ntsks, dtsks))
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task"):
                            a[i] = 1
                            for j in range(dtsks):
                                with openmp("task private(sum)"):
                                    sum = 0
                                    for k in range(10000):
                                        if k % 2 == 0:
                                            sum += 1
                                        else:
                                            sum -= 1
                                    da[i][j] = 1 + sum
                    with openmp("taskwait"):
                        ac = np.copy(a)
                        dac = np.copy(da)
                with openmp("barrier"):
                    pass
            return ac, dac

        r = test_impl(15, 10)
        np.testing.assert_array_equal(r[0], np.ones(r[0].shape))
        with self.assertRaises(AssertionError) as raises:
            np.testing.assert_array_equal(r[1], np.ones(r[1].shape))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_undeferred_task(self):
        @njit
        def test_impl():
            with openmp("parallel"):
                flag = 1
                with openmp("single"):
                    with openmp("task if(1) private(sum)"):
                        sum = 0
                        for i in range(10000):
                            if i % 2 == 0:
                                sum += 1
                            else:
                                sum -= 1
                        r = flag + sum
                    flag = 0
            return r

        assert test_impl()

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_untied_task_thread_assignment(self):
        @njit
        def test_impl(ntsks):
            start_nums = np.zeros(ntsks)
            current_nums = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task untied private(sum)"):
                            start_nums[i] = omp_get_thread_num()
                            with openmp("task if(0) shared(sum)"):
                                # Sleep
                                sum = 0
                                for j in range(10000):
                                    if j % 2 == 0:
                                        sum += 1
                                    else:
                                        sum -= 1
                            current_nums[i] = omp_get_thread_num() + sum
                with openmp("barrier"):
                    pass
            return start_nums, current_nums

        with self.assertRaises(AssertionError) as raises:
            sids, cids = test_impl(15)
            np.testing.assert_array_equal(sids, cids)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_taskyield_thread_assignment(self):
        @njit
        def test_impl(ntsks):
            start_nums = np.zeros(ntsks)
            finish_nums = np.zeros(ntsks)
            yielded_tasks = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task private(stn, start_i, finish_i, diff)"):
                            stn = omp_get_thread_num()
                            start_i = np.where(start_nums == stn)[0]
                            finish_i = np.where(finish_nums == stn)[0]
                            diff = np.zeros(len(start_i), dtype=np.int64)
                            for sindex in range(len(start_i)):
                                for findex in range(len(finish_i)):
                                    if start_i[sindex] == finish_i[findex]:
                                        break
                                else:
                                    diff[sindex] = start_i[sindex]
                            for dindex in diff[diff != 0]:
                                yielded_tasks[dindex] = 1
                            start_nums[i] = stn
                            with openmp("taskyield"):
                                pass
                            finish_nums[i] = omp_get_thread_num()
                with openmp("barrier"):
                    pass
            return yielded_tasks

        yt = test_impl(50)
        assert np.any(yt)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_final_task_thread_assignment(self):
        @njit
        def test_impl(ntsks, c):
            final_nums = np.zeros(ntsks)
            included_nums = np.zeros(ntsks)
            da = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task final(i>c) private(sum, d)"):
                            ftask_num = i
                            final_nums[ftask_num] = omp_get_thread_num()
                            # If it is a final task, generate an included task
                            if ftask_num > c:
                                d = 1
                                with openmp("task private(sum)"):
                                    itask_num = ftask_num
                                    # Sleep
                                    sum = 0
                                    for j in range(10000):
                                        if j % 2 == 0:
                                            sum += 1
                                        else:
                                            sum -= 1
                                    included_nums[itask_num] = omp_get_thread_num()
                                    da[itask_num] = d + sum
                                d = 0

            return final_nums, included_nums, da

        ntsks, c = 15, 5
        fns, ins, da = test_impl(ntsks, c)
        np.testing.assert_array_equal(fns[c:], ins[c:])
        np.testing.assert_array_equal(da, np.ones(ntsks))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_taskgroup(self):
        @njit
        def test_impl(ntsks, dtsks):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    with openmp("taskgroup"):
                        for i in range(ntsks):
                            with openmp("task"):
                                for _ in range(dtsks):
                                    with openmp("task"):
                                        # Sleep
                                        sum = 0
                                        for j in range(10000):
                                            if j % 2 == 0:
                                                sum += 1
                                            else:
                                                sum -= 1
                                        a[i] = 1 + sum
                    sa = np.copy(a)
            return a, sa

        ntsks = 15
        r = test_impl(ntsks, 10)
        np.testing.assert_array_equal(r[0], np.ones(ntsks))
        np.testing.assert_array_equal(r[1], np.ones(ntsks))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_task_priority(self):
        @njit
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            count = 0
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task priority(i)"):
                            count += i + 1
                            a[i] = count
            return a

        ntsks = 15
        r = test_impl(ntsks)
        rc = np.zeros(ntsks)
        for i in range(ntsks):
            rc[i] = sum(range(i + 1, ntsks + 1))
        np.testing.assert_array_equal(r, rc)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_task_mergeable(self):
        @njit
        def test_impl(ntsks, c1, c2):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task private(x)"):
                            x = c1
                            with openmp("task mergeable if(0)"):
                                x = c2
                            a[i] = x
            return a

        ntsks, c1, c2 = 75, 2, 3
        assert c2 in test_impl(ntsks, c1, c2)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_task_depend(self):
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            da = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task private(x, done)"):
                            x = 1
                            done = False
                            with openmp("task shared(x) depend(out: x)"):
                                x = 5
                            with openmp("""task shared(done, x)
                                        depend(out: done) depend(inout: x)"""):
                                x += i
                                done = True
                            with openmp("""task shared(done, x)
                                         depend(in: done) depend(inout: x)"""):
                                x *= i
                                da[i] = 1 if done else 0
                            with openmp("task shared(x) depend(in: x)"):
                                a[i] = x
            return a, da

        r = test_impl(15)
        # create check

    # Affinity clause should not affect result
    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_task_affinity(self):
        def test_impl(ntsks, const):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task firstprivate(i)"):
                            with openmp("""task shared(b) depend(out: b)
                                         affinity(a)"""):
                                b = np.full(i, const)
                            with openmp("""task shared(b) depend(in: b)
                                         affinity(a)"""):
                                a[i] = np.sum(b)
            return a

        test_impl(15, 4)
        # create check

    # What does this test?
    def test_shared_array(self):
        @njit
        def test_impl(mode):
            b = np.zeros(100)
            if mode == 0:
                return b

            with openmp("parallel"):
                with openmp("single"):
                    a = np.ones(100)
                    c = 0
                    d = 0
                    if mode > 1:
                        with openmp("task shared(a, c)"):
                            c = a.sum()
                        with openmp("task shared(a, d)"):
                            d = a.sum()
                        with openmp("taskwait"):
                            b[:] = c + d

            return b

        r = test_impl(0)
        np.testing.assert_array_equal(r, np.zeros(100))
        r = test_impl(1)
        np.testing.assert_array_equal(r, np.zeros(100))
        r = test_impl(2)
        np.testing.assert_array_equal(r, np.full(100, 200.0))


@unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
class TestOpenmpTaskloop(TestOpenmpBase):
    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_taskloop_basic(self):
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    with openmp("taskloop"):
                        for i in range(ntsks):
                            a[i] = 1
            return a

        r = test_impl(15)
        # create check

    def test_taskloop_num_tasks(self):
        @njit
        def test_impl(nt, iters, ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel num_threads(nt)"):
                with openmp("single"):
                    with openmp("taskloop num_tasks(ntsks)"):
                        for i in range(iters):
                            a[i] = omp_get_thread_num()
            return a

        nt, iters, ntsks = 8, 10, 4
        assert len(np.unique(test_impl(nt, iters, ntsks))) <= ntsks

    def test_taskloop_grainsize(self):
        @njit
        def test_impl(nt, iters, ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel num_threads(nt)"):
                with openmp("single"):
                    iters_per_task = iters // ntsks
                    with openmp("taskloop grainsize(iters_per_task)"):
                        for i in range(iters):
                            a[i] = omp_get_thread_num()
            return a

        nt, iters, ntsks = 8, 10, 4
        assert len(np.unique(test_impl(nt, iters, ntsks))) <= ntsks

    def test_taskloop_nogroup(self):
        @njit
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            sa = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    s = 0
                    with openmp("taskloop nogroup num_tasks(ntsks)"):
                        for i in range(ntsks):
                            a[i] = 1
                            sa[i] = s
                    with openmp("task priority(1)"):
                        s = 1
            return a, sa

        ntsks = 15
        r = test_impl(ntsks)
        np.testing.assert_array_equal(r[0], np.ones(ntsks))
        np.testing.assert_array_equal(r[1], np.ones(ntsks))

    def test_taskloop_collapse(self):
        @njit
        def test_impl(ntsks, nt):
            fl = np.zeros(ntsks)
            sl = np.zeros(ntsks)
            tl = np.zeros(ntsks)
            omp_set_num_threads(nt)
            with openmp("parallel"):
                with openmp("single"):
                    with openmp("taskloop collapse(2) num_tasks(ntsks)"):
                        for i in range(ntsks):
                            fl[i] = omp_get_thread_num()
                            for j in range(1):
                                sl[i] = omp_get_thread_num()
                                for k in range(1):
                                    tl[i] = omp_get_thread_num()

            return fl, sl, tl

        r = test_impl(25, 4)
        with self.assertRaises(AssertionError) as raises:
            np.testing.assert_array_equal(r[0], r[1])
        np.testing.assert_array_equal(r[1], r[2])


@linux_only
@unittest.skipUnless(
    TestOpenmpBase.skip_disabled or TestOpenmpBase.run_target, "Unimplemented"
)
class TestOpenmpTarget(TestOpenmpBase):
    """
    OpenMP target offloading tests. TEST_DEVICES is a required env var to
    specify the device numbers to run the tests on: 0 for host backend, 1 for
    CUDA backend. It is expected to be a comma-separated list of integer values.
    """

    devices = []
    assert TestOpenmpBase.test_devices, (
        "Expected env var TEST_DEVICES (comma-separated list of device numbers)"
    )
    devices = [int(devno) for devno in TestOpenmpBase.test_devices.split(",")]
    assert devices, "Expected non-empty test devices list"

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    @classmethod
    def is_testing_cpu(cls):
        return 1 in cls.devices

    # How to check for nowait?
    # Currently checks only compilation.
    # Numba optimizes the whole target away? This runs too fast.
    def target_nowait(self, device):
        target_pragma = f"target nowait device({device})"

        @njit
        def test_impl():
            with openmp(target_pragma):
                a = 0
                for i in range(1000000):
                    for j in range(1000000):
                        for k in range(1000000):
                            a += math.sqrt(i) + math.sqrt(j) + math.sqrt(k)

        test_impl()

    def target_nest_parallel_default_threadlimit(self, device):
        target_pragma = f"target device({device}) map(from: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                with openmp("parallel"):
                    teamno = omp_get_team_num()
                    threadno = omp_get_thread_num()
                    if teamno == 0 and threadno == 0:
                        teams = omp_get_num_teams()
                        threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        np.testing.assert_equal(teams, 1)
        self.assertGreater(threads, 1)

    def target_nest_parallel_set_numthreads(self, device):
        target_pragma = f"target device({device}) map(from: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                with openmp("parallel num_threads(32)"):
                    teamno = omp_get_team_num()
                    threadno = omp_get_thread_num()
                    if teamno == 0 and threadno == 0:
                        teams = omp_get_num_teams()
                        threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        np.testing.assert_equal(teams, 1)
        np.testing.assert_equal(threads, 32)

    def target_nest_teams_default_numteams(self, device):
        target_pragma = f"target device({device}) map(from: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                with openmp("teams"):
                    teamno = omp_get_team_num()
                    threadno = omp_get_thread_num()
                    if teamno == 0 and threadno == 0:
                        teams = omp_get_num_teams()
                        threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        # GPU device(0) starts >1 teams each with 1 thread.
        if device == 0:
            self.assertGreater(teams, 1)
            self.assertEqual(threads, 1)
        # CPU device(1) starts 1 team with >1 threads.
        elif device == 1:
            self.assertEqual(teams, 1)
            self.assertGreater(threads, 1)
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_nest_teams_set_numteams(self, device):
        target_pragma = f"target device({device}) map(from: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                with openmp("teams num_teams(32)"):
                    teamno = omp_get_team_num()
                    threadno = omp_get_thread_num()
                    if teamno == 0 and threadno == 0:
                        teams = omp_get_num_teams()
                        threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        if device == 0:
            self.assertEqual(teams, 32)
        elif device == 1:
            self.assertLessEqual(teams, 32)
        else:
            raise ValueError(f"Device {device} must be 0 or 1")
        self.assertGreaterEqual(threads, 1)

    def target_nest_teams_nest_parallel_default_numteams_threadlimit(self, device):
        target_pragma = f"target device({device}) map(from: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                with openmp("teams"):
                    with openmp("parallel"):
                        teamno = omp_get_team_num()
                        threadno = omp_get_thread_num()
                        if teamno == 0 and threadno == 0:
                            teams = omp_get_num_teams()
                            threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        # For GPU, impl. creates multiple threads and teams.
        if device == 0:
            self.assertGreater(teams, 1)
            self.assertGreater(threads, 1)
        # For CPU, impl. creates 1 teams with multiple threads.
        elif device == 1:
            self.assertEqual(teams, 1)
            self.assertGreater(threads, 1)
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_nest_teams_nest_parallel_set_numteams(self, device):
        target_pragma = f"target device({device}) map(from: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                with openmp("teams num_teams(32)"):
                    with openmp("parallel"):
                        teamno = omp_get_team_num()
                        threadno = omp_get_thread_num()
                        if teamno == 0 and threadno == 0:
                            teams = omp_get_num_teams()
                            threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        if device == 0:
            self.assertEqual(teams, 32)
        elif device == 1:
            self.assertGreaterEqual(teams, 1)
        else:
            raise ValueError(f"Device {device} must be 0 or 1")
        self.assertGreaterEqual(threads, 1)

    def target_nest_teams_nest_parallel_set_threadlimit(self, device):
        target_pragma = f"target device({device}) map(from: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                with openmp("teams thread_limit(32)"):
                    with openmp("parallel"):
                        teamno = omp_get_team_num()
                        threadno = omp_get_thread_num()
                        if teamno == 0 and threadno == 0:
                            teams = omp_get_num_teams()
                            threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        # For GPU, impl. creates > 1 teams.
        if device == 0:
            self.assertGreater(teams, 1)
            self.assertEqual(threads, 32)
        # For CPU, impl. creates exactly 1 team.
        elif device == 1:
            self.assertEqual(teams, 1)
            self.assertLessEqual(threads, 32)
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_nest_teams_nest_parallel_set_numteams_threadlimit(self, device):
        target_pragma = f"target device({device}) map(from: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                with openmp("teams num_teams(32) thread_limit(32)"):
                    with openmp("parallel"):
                        teamno = omp_get_team_num()
                        threadno = omp_get_thread_num()
                        if teamno == 0 and threadno == 0:
                            teams = omp_get_num_teams()
                            threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        self.assertGreaterEqual(teams, 1)
        if device == 0:
            self.assertEqual(teams, 32)
            self.assertEqual(threads, 32)
        elif device == 1:
            self.assertLessEqual(teams, 32)
            self.assertLessEqual(threads, 32)
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_nest_teams_nest_parallel_set_numteams_threadlimit_gt_numthreads(
        self, device
    ):
        target_pragma = f"target device({device}) map(from: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                with openmp("teams num_teams(32) thread_limit(64)"):
                    with openmp("parallel num_threads(32)"):
                        teamno = omp_get_team_num()
                        threadno = omp_get_thread_num()
                        if teamno == 0 and threadno == 0:
                            teams = omp_get_num_teams()
                            threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        self.assertGreaterEqual(teams, 1)
        if device == 0:
            self.assertEqual(teams, 32)
            self.assertEqual(threads, 32)
        elif device == 1:
            self.assertLessEqual(teams, 32)
            self.assertLessEqual(threads, 32)
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_nest_teams_nest_parallel_set_numteams_threadlimit_lt_numthreads(
        self, device
    ):
        target_pragma = f"target device({device}) map(from: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                # THREAD_LIMIT takes precedence over NUM_THREADS.
                with openmp("teams num_teams(32) thread_limit(64)"):
                    with openmp("parallel num_threads(128)"):
                        teamno = omp_get_team_num()
                        threadno = omp_get_thread_num()
                        if teamno == 0 and threadno == 0:
                            teams = omp_get_num_teams()
                            threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        self.assertGreaterEqual(teams, 1)
        if device == 0:
            self.assertEqual(teams, 32)
            self.assertEqual(threads, 64)
        elif device == 1:
            self.assertLessEqual(teams, 32)
            self.assertLessEqual(threads, 64)
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_nest_parallel_multiple_set_numthreads(self, device):
        target_pragma = (
            f"target device({device}) map(from: teams1, threads1, teams2, threads2)"
        )

        @njit
        def test_impl():
            teams1 = 0
            threads1 = 0
            teams2 = 0
            threads2 = 0
            with openmp(target_pragma):
                with openmp("parallel num_threads(32)"):
                    teamno = omp_get_team_num()
                    threadno = omp_get_thread_num()
                    if teamno == 0 and threadno == 0:
                        teams1 = omp_get_num_teams()
                        threads1 = omp_get_num_threads()
                with openmp("parallel num_threads(256)"):
                    teamno = omp_get_team_num()
                    threadno = omp_get_thread_num()
                    if teamno == 0 and threadno == 0:
                        teams2 = omp_get_num_teams()
                        threads2 = omp_get_num_threads()
            return teams1, threads1, teams2, threads2

        teams1, threads1, teams2, threads2 = test_impl()
        np.testing.assert_equal(teams1, 1)
        np.testing.assert_equal(threads1, 32)
        np.testing.assert_equal(teams2, 1)
        np.testing.assert_equal(threads2, 256)

    def target_nest_parallel_multiple_default_numthreads(self, device):
        target_pragma = (
            f"target device({device}) map(from: teams1, threads1, teams2, threads2)"
        )

        @njit
        def test_impl():
            teams1 = 0
            threads1 = 0
            teams2 = 0
            threads2 = 0
            with openmp(target_pragma):
                with openmp("parallel"):
                    teamno = omp_get_team_num()
                    threadno = omp_get_thread_num()
                    if teamno == 0 and threadno == 0:
                        teams1 = omp_get_num_teams()
                        threads1 = omp_get_num_threads()
                with openmp("parallel"):
                    teamno = omp_get_team_num()
                    threadno = omp_get_thread_num()
                    if teamno == 0 and threadno == 0:
                        teams2 = omp_get_num_teams()
                        threads2 = omp_get_num_threads()
            return teams1, threads1, teams2, threads2

        teams1, threads1, teams2, threads2 = test_impl()
        np.testing.assert_equal(teams1, 1)
        self.assertGreater(threads1, 1)
        np.testing.assert_equal(teams2, 1)
        self.assertGreater(threads2, 1)

    def target_nest_parallel_multiple_set_numthreads_byone(self, device):
        target_pragma = f"target device({device}) map(from: max_threads, teams1, threads1, teams2, threads2)"

        @njit
        def test_impl():
            max_threads = 0
            teams1 = 0
            threads1 = 0
            teams2 = 0
            threads2 = 0
            with openmp(target_pragma):
                max_threads = omp_get_max_threads()
                with openmp("parallel"):
                    teamno = omp_get_team_num()
                    threadno = omp_get_thread_num()
                    if teamno == 0 and threadno == 0:
                        teams1 = omp_get_num_teams()
                        threads1 = omp_get_num_threads()
                with openmp("parallel num_threads(256)"):
                    teamno = omp_get_team_num()
                    threadno = omp_get_thread_num()
                    if teamno == 0 and threadno == 0:
                        teams2 = omp_get_num_teams()
                        threads2 = omp_get_num_threads()
            return max_threads, teams1, threads1, teams2, threads2

        # NOTE: max_threads for device(0) is the number of threads set by the
        # sibling parallel legion with the highest num_threads clause.
        # For device(1), is the number of max threads as determined by the host
        # runtime.
        max_threads, teams1, threads1, teams2, threads2 = test_impl()
        np.testing.assert_equal(teams1, 1)
        np.testing.assert_equal(threads1, max_threads)
        np.testing.assert_equal(teams2, 1)
        np.testing.assert_equal(threads2, 256)

    def target_nest_parallel(self, device):
        # TODO: map should be "from" instead of "tofrom" once this is fixed.
        target_pragma = f"target device({device}) map(from: a)"
        # NOTE: num_threads should be a multiple of warp size, e.g. for NVIDIA
        # V100 it is 32, the OpenMP runtime floors non-multiple of warp size.
        # TODO: Newer LLVM versions should not have this restriction.
        parallel_pragma = (
            "parallel num_threads(32)"  # + (" shared(a)" if explicit else "")
        )

        @njit
        def test_impl():
            a = np.zeros(32, dtype=np.int64)
            with openmp(target_pragma):
                with openmp(parallel_pragma):
                    thread_id = omp_get_thread_num()
                    a[thread_id] = 1
            return a

        r = test_impl()
        np.testing.assert_equal(r, np.full(32, 1))

    def target_parallel_for_range_step_arg(self, device):
        target_pragma = f"target device({device}) map(tofrom: a)"
        parallel_pragma = "parallel for"
        N = 10
        step = 2

        @njit
        def test_impl():
            a = np.zeros(N, dtype=np.int32)
            with openmp(target_pragma):
                with openmp(parallel_pragma):
                    for i in range(0, len(a), step):
                        a[i] = i + 1

            return a

        r = test_impl()
        np.testing.assert_equal(r, np.array([1, 0, 3, 0, 5, 0, 7, 0, 9, 0]))

    def target_parallel_for_incremented_step(self, device):
        target_pragma = f"target device({device}) map(tofrom: a)"
        parallel_pragma = "parallel for"
        N = 10
        step_range = 3

        @njit
        def test_impl():
            a = np.zeros(N, dtype=np.int32)
            for i in range(step_range):
                with openmp(target_pragma):
                    with openmp(parallel_pragma):
                        for j in range(0, len(a), i + 1):
                            a[j] = i + 1
            return a

        r = test_impl()
        np.testing.assert_equal(r, np.array([3, 1, 2, 3, 2, 1, 3, 1, 2, 3]))

    def target_teams(self, device):
        target_pragma = (
            f"target teams num_teams(100) device({device}) map(from: a, nteams)"
        )

        @njit
        def test_impl():
            a = np.zeros(100, dtype=np.int64)
            nteams = 0
            with openmp(target_pragma):
                team_id = omp_get_team_num()
                if team_id == 0:
                    nteams = omp_get_num_teams()
                a[team_id] = 1
            return a, nteams

        r, nteams = test_impl()
        if device == 0:
            np.testing.assert_equal(r, np.full(100, 1))
        elif device == 1:
            np.testing.assert_equal(r[:nteams], np.full(nteams, 1))
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_nest_teams(self, device):
        target_pragma = f"target device({device}) map(from: a, nteams)"

        @njit
        def test_impl():
            a = np.zeros(100, dtype=np.int64)
            nteams = 0
            with openmp(target_pragma):
                with openmp("teams num_teams(100)"):
                    team_id = omp_get_team_num()
                    if team_id == 0:
                        nteams = omp_get_num_teams()
                    a[team_id] = 1
            return a, nteams

        r, nteams = test_impl()
        if device == 0:
            np.testing.assert_equal(r, np.full(100, 1))
        elif device == 1:
            np.testing.assert_equal(r[:nteams], np.full(nteams, 1))
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_nest_teams_from_shared_expl_scalar(self, device):
        target_pragma = f"target device({device}) map(from: s)"

        @njit
        def test_impl():
            s = 0
            with openmp(target_pragma):
                with openmp("teams num_teams(100) shared(s)"):
                    team_id = omp_get_team_num()
                    if team_id == 0:
                        s = 1
            return s

        s = test_impl()
        np.testing.assert_equal(s, 1)

    def target_nest_teams_from_shared_impl_scalar(self, device):
        target_pragma = f"target device({device}) map(from: s)"

        @njit
        def test_impl():
            s = 0
            with openmp(target_pragma):
                with openmp("teams num_teams(100)"):
                    team_id = omp_get_team_num()
                    if team_id == 0:
                        s = 1
            return s

        s = test_impl()
        np.testing.assert_equal(s, 1)

    def target_nest_teams_tofrom_shared_expl_scalar(self, device):
        target_pragma = f"target device({device}) map(tofrom: s)"

        @njit
        def test_impl():
            s = 0
            with openmp(target_pragma):
                with openmp("teams num_teams(100) shared(s)"):
                    team_id = omp_get_team_num()
                    if team_id == 0:
                        s = 1
            return s

        s = test_impl()
        np.testing.assert_equal(s, 1)

    def target_nest_teams_tofrom_shared_impl_scalar(self, device):
        target_pragma = f"target device({device}) map(tofrom: s)"

        @njit
        def test_impl():
            s = 0
            ss = np.zeros(1)
            with openmp(target_pragma):
                with openmp("teams num_teams(100)"):
                    team_id = omp_get_team_num()
                    if team_id == 0:
                        s = 1
                        ss[0] = 1
            return s, ss

        s, ss = test_impl()
        np.testing.assert_equal(s, 1)
        np.testing.assert_equal(ss, 1)

    def target_teams_nest_parallel(self, device):
        target_pragma = f"target teams device({device}) num_teams(10) thread_limit(32) map(tofrom: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                with openmp("parallel"):
                    team_id = omp_get_team_num()
                    thread_id = omp_get_thread_num()
                    if team_id == 0 and thread_id == 0:
                        teams = omp_get_num_teams()
                        threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        if device == 0:
            self.assertEqual(teams, 10)
            self.assertEqual(threads, 32)
        elif device == 1:
            self.assertLessEqual(teams, 10)
            self.assertLessEqual(threads, 32)
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_teams_nest_parallel_set_thread_limit(self, device):
        target_pragma = f"target device({device}) map(tofrom: teams, threads)"

        @njit
        def test_impl():
            teams = 0
            threads = 0
            with openmp(target_pragma):
                with openmp("teams num_teams(10) thread_limit(32)"):
                    with openmp("parallel"):
                        team_id = omp_get_team_num()
                        thread_id = omp_get_thread_num()
                        if team_id == 0 and thread_id == 0:
                            teams = omp_get_num_teams()
                            threads = omp_get_num_threads()
            return teams, threads

        teams, threads = test_impl()
        if device == 0:
            self.assertEqual(teams, 10)
            self.assertEqual(threads, 32)
        elif device == 1:
            self.assertLessEqual(teams, 10)
            self.assertLessEqual(threads, 32)
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_map_to_scalar(self, device):
        target_pragma = f"target device({device}) map(to: x) map(from: r)"

        @njit
        def test_impl(x):
            with openmp(target_pragma):
                x += 1
                r = x
            return r

        x = 42
        r = test_impl(x)
        np.testing.assert_equal(r, 43)

    def target_map_to_array(self, device):
        target_pragma = f"target device({device}) map(to: a) map(from: r)"

        @njit
        def test_impl(a):
            with openmp(target_pragma):
                r = 0
                for i in range(len(a)):
                    r += a[i]
            return r

        n = 10
        a = np.ones(n)
        r = test_impl(a)
        # r is the sum of array elements (ones-array), thus must equal s.
        np.testing.assert_equal(r, n)

    def target_map_from_scalar(self, device):
        target_pragma = f"target device({device}) map(from: x)"

        @njit
        def test_impl(x):
            with openmp(target_pragma):
                x = 43
            return x

        x = 42
        r = test_impl(x)
        np.testing.assert_equal(r, 43)

    def target_map_tofrom_scalar(self, device):
        target_pragma = f"target device({device}) map(tofrom: x)"

        @njit
        def test_impl(x):
            with openmp(target_pragma):
                x += 1
            return x

        x = 42
        r = test_impl(x)
        np.testing.assert_equal(r, 43)

    def target_multiple_map_tofrom_scalar(self, device):
        target_pragma = f"target device({device}) map(tofrom: x)"

        @njit
        def test_impl(x):
            with openmp(target_pragma):
                x += 1
            with openmp(target_pragma):
                x += 1
            return x

        x = 42
        r = test_impl(x)
        np.testing.assert_equal(r, 44)

    def target_map_from_array(self, device):
        target_pragma = f"target device({device}) map(from: a)"

        @njit
        def test_impl(n):
            a = np.zeros(n, dtype=np.int64)
            with openmp(target_pragma):
                for i in range(len(a)):
                    a[i] = 42
            return a

        n = 10
        r = test_impl(n)
        np.testing.assert_array_equal(r, np.full(n, 42))

    def target_map_slice_in_mapping(self, device):
        target_pragma = f"target device({device}) map(a[50:100]) map(to: b[100:150])"

        @njit
        def test_impl(n):
            a = np.zeros(n)
            b = np.arange(n)
            with openmp(target_pragma):
                for i in range(50):
                    # These b accesses are within the transferred region.
                    a[i + 50] = b[i + 100]
            return a

        n = 200
        r = test_impl(n)
        np.testing.assert_array_equal(r[0:50], np.zeros(50))
        np.testing.assert_array_equal(r[50:100], np.arange(n)[100:150])
        np.testing.assert_array_equal(r[100:200], np.zeros(100))

    def target_map_slice_read_out_mapping(self, device):
        target_pragma = f"target device({device}) map(a[50:100]) map(to: b[100:150])"

        @njit
        def test_impl(n):
            a = np.zeros(n)
            b = np.arange(n)
            with openmp(target_pragma):
                for i in range(50):
                    # These b accesses are outside the transferred region.
                    # Should get whatever happens to be in memory at that point.
                    # We assume that isn't arange(50:100).
                    a[i + 50] = b[i + 50]
            return a

        n = 200
        r = test_impl(n)
        np.testing.assert_array_equal(r[0:50], np.zeros(50))
        # Make sure that the range 50-100 was not transferred.
        assert not np.array_equal(r[50:100], np.arange(n)[50:100])
        np.testing.assert_array_equal(r[100:200], np.zeros(100))

    def target_map_tofrom_array(self, device):
        target_pragma = f"target device({device}) map(tofrom: a)"

        @njit
        def test_impl(a):
            with openmp(target_pragma):
                for i in range(len(a)):
                    a[i] += 1
            return a

        n = 10
        a = np.full(n, 42)
        r = test_impl(a)
        np.testing.assert_array_equal(r, np.full(n, 43))

    def target_nest_parallel_for(self, device):
        target_pragma = f"target device({device}) map(tofrom: a, sched)"

        @njit
        def test_impl(a, sched):
            with openmp(target_pragma):
                with openmp("parallel for num_threads(256)"):
                    for i in range(len(a)):
                        a[i] = 1
                        thread_id = omp_get_thread_num()
                        sched[i] = thread_id
            return a, sched

        n = 1000
        a = np.zeros(n)
        sched = np.zeros(n)
        r, sched = test_impl(a, sched)
        np.testing.assert_array_equal(r, np.ones(n))
        # u = unique thread ids that processed the array, c = number of iters
        # each unique thread id has processed.
        u, c = np.unique(sched, return_counts=True)
        # test that 256 threads executed.
        np.testing.assert_equal(len(u), 256)
        # test that each thread executed more than 1 iteration.
        for ci in c:
            self.assertGreater(ci, 0)

    def target_nest_teams_distribute(self, device):
        target_pragma = f"target device({device}) map(tofrom: a, sched)"

        @njit
        def test_impl(a, sched):
            with openmp(target_pragma):
                with openmp("teams distribute"):
                    for i in range(len(a)):
                        a[i] = 1
                        team_id = omp_get_team_num()
                        sched[i] = team_id
            return a, sched

        n = 100
        a = np.zeros(n)
        sched = np.zeros(n)
        r, sched = test_impl(a, sched)
        np.testing.assert_array_equal(r, np.ones(n))
        # u = unique teams ids that processed the array, c = number of iters
        # each unique team id has processed.
        u, c = np.unique(sched, return_counts=True)
        if device == 0:
            # For GPU, OpenMP creates as many teams as the number of iterations,
            # where each team leader executes one iteration.
            np.testing.assert_equal(len(u), n)
            np.testing.assert_array_equal(c, np.ones(n))
        elif device == 1:
            # For CPU, OpenMP creates 1 teams with 1 thread processing all n
            # iterations.
            np.testing.assert_equal(len(u), 1)
            np.testing.assert_array_equal(c, [100])
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_teams_distribute(self, device):
        target_pragma = (
            f"target teams distribute device({device}) map(tofrom: a, sched)"
        )

        @njit
        def test_impl(a, sched):
            with openmp(target_pragma):
                for i in range(len(a)):
                    a[i] = 1
                    team_id = omp_get_team_num()
                    sched[i] = team_id
            return a, sched

        n = 1000
        a = np.zeros(n)
        sched = np.zeros(n)
        r, sched = test_impl(a, sched)
        np.testing.assert_array_equal(r, np.ones(n))
        # u = unique teams ids that processed the array, c = number of iters
        # each unique team id has processed.
        u, c = np.unique(sched, return_counts=True)
        if device == 0:
            # For GPU, impl. creates as many teams as the number of iterations,
            # where each team leader executes one iteration.
            np.testing.assert_equal(len(u), n)
            np.testing.assert_array_equal(c, np.ones(n))
        elif device == 1:
            # For CPU, impl. creates 1 team which processes all iterations.
            np.testing.assert_equal(len(u), 1)
            np.testing.assert_array_equal(c, [1000])
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_teams_distribute_set_num_teams(self, device):
        target_pragma = (
            f"target teams distribute device({device}) map(tofrom: a) num_teams(4)"
        )

        @njit
        def test_impl(a, sched):
            with openmp(target_pragma):
                for i in range(len(a)):
                    a[i] = 1
                    team_id = omp_get_team_num()
                    sched[i] = team_id
            return a, sched

        n = 1000
        a = np.zeros(n)
        sched = np.zeros(n)
        r, sched = test_impl(a, sched)
        np.testing.assert_array_equal(r, np.ones(n))
        # u = unique teams ids that processed the array, c = number of iters
        # each unique team id has processed.
        u, c = np.unique(sched, return_counts=True)
        np.testing.assert_equal(len(u), 4)
        np.testing.assert_array_equal(c, np.full(4, 250))

    def target_firstprivate_scalar_explicit(self, device):
        target_pragma = f"target device({device}) firstprivate(s)"

        @njit
        def test_impl(s):
            with openmp(target_pragma):
                s = 43
            return s

        s = 42
        r = test_impl(s)
        np.testing.assert_equal(r, 42)

    def target_firstprivate_scalar_implicit(self, device):
        target_pragma = f"target device({device})"

        @njit
        def test_impl(s):
            with openmp(target_pragma):
                s = 43
            return s

        s = 42
        r = test_impl(s)
        np.testing.assert_equal(r, 42)

    def target_data_from(self, device):
        target_data_pragma = f"""target data device({device})
                                map(from: a)"""
        target_pragma = f"target device({device})"

        @njit
        def test_impl():
            a = np.ones(10)
            with openmp(target_data_pragma):
                with openmp(target_pragma):
                    for i in range(len(a)):
                        a[i] = 42
            return a

        a = test_impl()
        np.testing.assert_array_equal(a, np.full(10, 42))

    def target_data_to(self, device):
        target_data_pragma = f"""target data device({device})
                                map(to: a) map(from: b)"""
        target_pragma = f"target device({device})"

        @njit
        def test_impl():
            a = np.ones(10)
            b = np.zeros(10)
            with openmp(target_data_pragma):
                with openmp(target_pragma):
                    for i in range(len(a)):
                        a[i] = 42
                        b[i] = a[i]
            return a, b

        a, b = test_impl()
        np.testing.assert_array_equal(a, np.ones(10))
        np.testing.assert_array_equal(b, np.full(10, 42))

    def target_data_tofrom(self, device):
        target_data_pragma = f"""target data device({device})
                                map(tofrom: s, a)"""
        target_pragma = f"target device({device})"

        @njit
        def test_impl():
            s = 0
            a = np.ones(10)
            with openmp(target_data_pragma):
                with openmp(target_pragma):
                    for i in range(len(a)):
                        a[i] += 41
                    s = 42
            return s, a

        s, a = test_impl()
        # s is a FIRSTPRIVATE in the target region, so changes do not affect
        # host s despite FROM mapping.
        np.testing.assert_equal(s, 0)
        np.testing.assert_array_equal(a, np.full(10, 42))

    def target_data_alloc_from(self, device):
        target_data_pragma = f"""target data device({device})
                                map(alloc: a) map(from: b)"""
        target_pragma = f"target device({device})"

        @njit
        def test_impl():
            a = np.ones(10)
            b = np.zeros(10)
            with openmp(target_data_pragma):
                with openmp(target_pragma):
                    for i in range(len(a)):
                        a[i] = 42
                        b[i] = a[i]
            return a, b

        a, b = test_impl()
        np.testing.assert_array_equal(a, np.ones(10))
        np.testing.assert_array_equal(b, np.full(10, 42))

    def target_data_mix_to_from(self, device):
        target_data_pragma = f"""target data device({device})
                                map(to: a) map(from: b)"""
        target_pragma = f"target device({device})"

        @njit
        def test_impl():
            a = np.ones(10)
            b = np.ones(10)
            with openmp(target_data_pragma):
                with openmp(target_pragma):
                    for i in range(len(a)):
                        a[i] = 42
                        b[i] = 42
            return a, b

        a, b = test_impl()
        np.testing.assert_array_equal(a, np.ones(10))
        np.testing.assert_array_equal(b, np.full(10, 42))

    def target_update_from(self, device):
        target_data_pragma = f"""target data device({device})
                                map(to: a)"""
        target_pragma = f"target device({device})"
        target_update_pragma = f"target update from(a) device({device})"

        @njit
        def test_impl():
            a = np.ones(10)
            with openmp(target_data_pragma):
                with openmp(target_pragma):
                    for i in range(len(a)):
                        a[i] = 42
                with openmp(target_update_pragma):
                    pass
            return a

        a = test_impl()
        np.testing.assert_array_equal(a, np.full(10, 42))

    def target_update_to(self, device):
        target_data_pragma = f"""target data device({device})
                                map(from: a)"""
        target_pragma = f"target device({device})"
        target_update_pragma = f"target update to(a) device({device})"

        @njit
        def test_impl():
            a = np.ones(10)
            with openmp(target_data_pragma):
                a += 1

                with openmp(target_update_pragma):
                    pass

                with openmp(target_pragma):
                    for i in range(len(a)):
                        a[i] += 1
            return a

        a = test_impl()
        np.testing.assert_array_equal(a, np.full(10, 3))

    def target_update_to_from(self, device):
        target_data_pragma = f"""target data device({device})
                                map(to: a)"""
        target_pragma = f"target device({device})"
        target_update_to_pragma = f"target update to(a) device({device})"
        target_update_from_pragma = f"target update from(a) device({device})"

        @njit
        def test_impl():
            a = np.ones(10)
            with openmp(target_data_pragma):
                a += 1

                with openmp(target_update_to_pragma):
                    pass

                with openmp(target_pragma):
                    for i in range(len(a)):
                        a[i] += 1

                with openmp(target_update_from_pragma):
                    pass

                a += 1
            return a

        a = test_impl()
        np.testing.assert_array_equal(a, np.full(10, 4))

    # WEIRD: breaks when runs alone, passes if runs with all tests.
    def target_enter_exit_data_to_from_hostonly(self, device):
        target_enter = f"""target enter data device({device})
                                map(to: a)"""

        target_exit = f"""target exit data device({device})
                                map(from: a)"""

        @njit
        def test_impl():
            a = np.ones(10)
            with openmp(target_enter):
                pass

            a += 1

            # XXX: Test passes if uncommented!
            # with openmp("target device(1)"):
            #    pass

            with openmp(target_exit):
                pass

            return a

        a = test_impl()
        np.testing.assert_array_equal(a, np.full(10, 1))

    # WEIRD: breaks when runs alone, passes if runs with all tests.
    def target_data_tofrom_hostonly(self, device):
        target_data = f"""target data device({device})
                                map(tofrom: a)"""

        @njit
        def test_impl():
            a = np.ones(10)
            with openmp(target_data):
                a += 1

            # XXX: Test passes if uncommented!
            # with openmp("target device(1)"):
            #    pass

            return a

        a = test_impl()
        np.testing.assert_array_equal(a, np.full(10, 1))

    def target_data_update(self, device):
        target_pragma = f"target teams distribute parallel for device({device})"
        target_data = f"target data map(from:a) device({device})"
        target_update = f"target update to(a) device({device})"

        @njit
        def test_impl(a):
            with openmp(target_data):
                for rep in range(10):
                    # Target update resets a to ones.
                    with openmp(target_update):
                        pass
                    with openmp(target_pragma):
                        for i in range(len(a)):
                            a[i] += 1

        a = np.ones(4)
        test_impl(a)
        np.testing.assert_array_equal(a, np.full(4, 2))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Abort - unimplemented")
    def target_data_nest_multiple_target(self, device):
        target_data_pragma = f"""target data device({device}) map(to: a)
                        map(tofrom: b) map(from: as1, as2, bs1, bs2)"""
        target_pragma = f"target device({device})"

        @njit
        def test_impl(s, n1, n2):
            a = np.full(s, n1)
            as1 = np.empty(s, dtype=a.dtype)
            as2 = np.empty(s, dtype=a.dtype)
            b = n1
            with openmp(target_data_pragma):
                with openmp(target_pragma):
                    as1[:] = a
                    bs1 = b
                with openmp(target_pragma):
                    for i in range(s):
                        a[i] = n2
                    b = n2
                with openmp(target_pragma):
                    as2[:] = a
                    bs2 = b
            return a, as1, as2, b, bs1, bs2

        s, n1, n2 = 50, 1, 2
        ao, a1, a2, bo, b1, b2 = test_impl(s, n1, n2)
        np.testing.assert_array_equal(ao, np.full(s, n1))
        np.testing.assert_array_equal(a1, np.full(s, n1))
        np.testing.assert_array_equal(a2, np.full(s, n2))
        assert bo == n2
        assert b1 == n1
        assert b2 == n2

    @unittest.skip("Creates map entries that aren't cleared.")
    def target_enter_exit_data_array_sections(self, device):
        target_enter_pragma = (
            f"target enter data map(to: a[0:3], b[bstart:bstop]) device({device})"
        )
        target_exit_pragma = f"target exit data map(from: a[0:3]) device({device})"
        target_pragma = f"target teams distribute parallel for device({device})"

        @njit
        def test_impl():
            bstart = 0
            bstop = 3
            a = np.array([1, 2, 3])
            b = np.array([3, 2, 1])
            with openmp(target_enter_pragma):
                with openmp(target_pragma):
                    for i in range(1):
                        a[0] = 42
                        b[0] = 42

            with openmp(target_exit_pragma):
                pass

            return a, b

        a, b = test_impl()
        np.testing.assert_array_equal(a, [42, 2, 3])
        np.testing.assert_array_equal(b, [3, 2, 1])

    def target_enter_exit_data(self, device):
        target_enter_pragma = f"""target enter data device({device})
                            map(to: scalar) map(to: array)"""
        target_exit_pragma = f"""target exit data device({device})
                            map(from: scalar, array)"""
        target_pragma = f"target device({device})"

        @njit
        def test_impl(scalar, array):
            with openmp(target_enter_pragma):
                pass

            with openmp(target_pragma):
                scalar += 1
                for i in range(len(array)):
                    array[i] += 1

            with openmp(target_exit_pragma):
                pass

            return scalar, array

        n = 10
        s = 42
        a = np.full(n, 42)
        r_s, r_a = test_impl(s, a)
        # NOTE: This is confusing but spec compliant and matches OpenMP target
        # offloading of the C/C++ version: scalar is implicitly a firstprivate
        # thus it does not copy back to the host although it is in a "from" map
        # of the target exit data directive.

        # TODO: we may want to revise Python behavior and copy back scalar too.
        np.testing.assert_equal(r_s, 42)
        np.testing.assert_array_equal(r_a, np.full(n, 43))

    def target_enter_exit_data_alloc(self, device):
        target_enter_pragma = f"""target enter data device({device})
                                map(alloc: a)"""
        target_exit_pragma = f"target exit data device({device}) map(from: a)"
        target_pragma = f"target device({device})"

        @njit
        def test_impl(a):
            with openmp(target_enter_pragma):
                pass
            with openmp(target_pragma):
                for i in range(len(a)):
                    a[i] = 1
            with openmp(target_exit_pragma):
                pass

            return a

        n = 100
        a = np.zeros(n)
        r = test_impl(a)
        np.testing.assert_array_equal(r, np.ones(n))

    def target_teams_distribute_parallel_for(self, device):
        target_pragma = f"""target teams distribute parallel for
                        device({device}) num_teams(4)
                        map(tofrom: s, a, sched_team, sched_thread)"""

        @njit
        def test_impl(a, sched_team, sched_thread):
            s = 42
            with openmp(target_pragma):
                for i in range(len(a)):
                    a[i] = 1
                    team_id = omp_get_team_num()
                    sched_team[i] = team_id
                    thread_id = omp_get_thread_num()
                    sched_thread[i] = thread_id
                    if i == 0 and team_id == 0 and thread_id == 0:
                        s += 1
            return s, a, sched_team, sched_thread

        n = 1024
        a = np.zeros(n)
        sched_team = np.zeros(n)
        sched_thread = np.zeros(n)
        s, r, sched_team, sched_thread = test_impl(a, sched_team, sched_thread)
        self.assertEqual(s, 43)
        np.testing.assert_array_equal(r, np.ones(n))
        # u_team stores unique ids of teams, c_team stores how many iterations
        # each time executed.
        u_team, c_team = np.unique(sched_team, return_counts=True)
        # u_thread stores unique ids of threads (regardless of team), c_thread
        # stores how many iterations threads of the same unique id executed.
        u_thread, c_thread = np.unique(sched_thread, return_counts=True)
        if device == 0:
            # there are 4 teams each with a unique id starting from 0.
            self.assertEqual(len(u_team), 4)
            np.testing.assert_array_equal(u_team, np.arange(0, len(u_team)))
            # each team should execute 1024/4 = 256 iterations.
            np.testing.assert_array_equal(c_team, np.full(len(c_team), n / len(u_team)))
            # Expect equal number of iterations per thread id across teams.
            np.testing.assert_array_equal(
                c_thread, np.full(len(u_thread), n / len(u_thread))
            )
        elif device == 1:
            self.assertLessEqual(len(u_team), 4)
            np.testing.assert_array_equal(u_team, np.arange(0, len(u_team)))
            # Divide (integer) n iterations by number of teams and add the
            # remainder.
            chunk = n // len(u_team)
            rem = n % len(u_team)
            chunks = np.full(len(u_team), chunk)
            chunks[:rem] += 1
            np.testing.assert_array_equal(c_team, chunks)

            # Divide (integer) per team iterations by number of threads and add the
            # remainder.
            chunks_thread = np.zeros(len(u_thread))
            for i in range(len(u_team)):
                chunk = chunks[i] // len(u_thread)
                rem = chunks[i] % len(u_thread)
                chunk_thread = np.full(len(u_thread), chunk)
                chunk_thread[:rem] += 1
                chunks_thread += chunk_thread

            np.testing.assert_array_equal(c_thread, chunks_thread)
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    @unittest.skip("Fix unexpected QUAL.OMP.THREAD_LIMIT")
    def target_teams_nest_distribute_parallel_for(self, device):
        target_pragma = f"""target teams device({device}) num_teams(4)
                        map(tofrom: s, a, sched_team, sched_thread)"""
        dist_parfor_pragma = "distribute parallel for num_threads(256)"

        @njit
        def test_impl(a, sched_team, sched_thread):
            s = 42
            with openmp(target_pragma):
                with openmp(dist_parfor_pragma):
                    for i in range(len(a)):
                        a[i] = 1
                        team_id = omp_get_team_num()
                        sched_team[i] = team_id
                        thread_id = omp_get_thread_num()
                        sched_thread[i] = thread_id
                        if i == 0 and team_id == 0 and thread_id == 0:
                            s += 1
            return s, a, sched_team, sched_thread

        n = 1024
        a = np.zeros(n)
        sched_team = np.zeros(n)
        sched_thread = np.zeros(n)
        s, r, sched_team, sched_thread = test_impl(a, sched_team, sched_thread)
        np.testing.assert_equal(s, 43)
        np.testing.assert_array_equal(r, np.ones(n))
        u_team, c_team = np.unique(sched_team, return_counts=True)
        # there are 4 teams each with a unique id starting from 0.
        np.testing.assert_equal(len(u_team), 4)
        np.testing.assert_array_equal(u_team, np.arange(0, len(u_team)))
        # each team should execute 1024/4 = 256 iterations.
        np.testing.assert_array_equal(c_team, np.full(len(c_team), n / len(u_team)))
        u_thread, c_thread = np.unique(sched_thread, return_counts=True)
        # testing thread scheduling is tricky: OpenMP runtime sets aside a warp
        # for the "sequential" target region execution.
        # TODO: update tests as newer LLVM version lift the above limitations.
        self.assertGreaterEqual(len(u_thread), n / len(u_team) - 32)
        for c_thread_i in c_thread:
            # threads from team 0 will execute more iterations (see above
            # comment on removed warp).
            self.assertGreaterEqual(c_thread_i, 4)

    def target_teams_nest_parallel_fpriv_shared_scalar(self, device):
        target_pragma = f"target teams num_teams(1) thread_limit(32) device({device}) map(from: threads)"

        @njit
        def test_impl():
            s = 42
            r = np.zeros(32)
            threads = 0
            with openmp(target_pragma):
                with openmp("parallel firstprivate(s)"):
                    threadno = omp_get_thread_num()
                    if threadno == 0:
                        threads = omp_get_num_threads()
                    s += 1
                    r[threadno] = s
            return s, r, threads

        s, r, threads = test_impl()
        self.assertEqual(s, 42)
        self.assertLessEqual(threads, 32)
        np.testing.assert_array_equal(r[:threads], np.full(threads, 43))

    def target_nest_parallel_float_fpriv(self, device):
        target_pragma = f"target device({device}) map(from: r)"

        @njit
        def test_impl():
            s = np.float32(42.0)
            r = np.float32(0.0)
            with openmp(target_pragma):
                with openmp("parallel firstprivate(s)"):
                    threadno = omp_get_thread_num()
                    if threadno == 0:
                        r = s + 1
            return r

        r = test_impl()
        np.testing.assert_equal(r, 43.0)

    def target_nest_teams_float_fpriv(self, device):
        target_pragma = f"target device({device}) map(from: r)"

        @njit
        def test_impl():
            s = np.float32(42.0)
            r = np.float32(0.0)
            with openmp(target_pragma):
                with openmp("teams firstprivate(s)"):
                    teamno = omp_get_thread_num()
                    if teamno == 0:
                        r = s + 1
            return r

        r = test_impl()
        np.testing.assert_equal(r, 43.0)

    @unittest.skip("Frontend codegen error")
    def target_teams_nest_parallel_fpriv_shared_array(self, device):
        target_pragma = f"target teams num_teams(1) thread_limit(32) device({device})"

        # FIX: frontend fails to emit copy constructor, error:
        # add_llvm_module is not supported on the CUDACodelibrary
        # QUESTION: in which address space does the copy constructor create the copy on the GPU?
        @njit
        def test_impl():
            s = np.zeros(32)
            with openmp(target_pragma):
                with openmp("parallel firstprivate(s)"):
                    print("parallel s", s[0])
                    teams = omp_get_num_teams()
                    threads = omp_get_num_threads()
                    teamno = omp_get_team_num()
                    threadno = omp_get_thread_num()
                    if teamno == 0 and threadno == 0:
                        print("teams", teams, "threads", threads)

        test_impl()
        input("ok?")

    def target_teams_shared_array(self, device):
        target_pragma = f"target teams num_teams(10) map(tofrom: a) map(from: nteams) device({device})"

        @njit
        def test_impl():
            a = np.zeros(10, dtype=np.int32)
            nteams = 0

            with openmp(target_pragma):
                team_shared_array = np.empty(10, dtype=np.int32)
                team_id = omp_get_team_num()

                if team_id == 0:
                    nteams = omp_get_num_teams()

                for i in range(10):
                    team_shared_array[i] = team_id

                lasum = 0
                for i in range(10):
                    lasum += team_shared_array[i]
                a[team_id] = lasum

            return a, nteams

        r, nteams = test_impl()
        expected = np.arange(10) * 10
        if device == 0:
            np.testing.assert_array_equal(r, expected)
        elif device == 1:
            np.testing.assert_array_equal(r[:nteams], expected[:nteams])
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_teams_shared_array_2d(self, device):
        target_pragma = f"target teams num_teams(10) map(tofrom: a) map(from: nteams) device({device})"

        @njit
        def test_impl():
            a = np.zeros((10, 2, 2), dtype=np.int32)
            nteams = 0

            with openmp(target_pragma):
                team_shared_array = np.empty((2, 2), dtype=np.int32)
                team_id = omp_get_team_num()

                if team_id == 0:
                    nteams = omp_get_num_teams()

                for i in range(2):
                    for j in range(2):
                        team_shared_array[i, j] = team_id

                for i in range(2):
                    for j in range(2):
                        a[team_id, i, j] = team_shared_array[i, j]
            return a, nteams

        a, nteams = test_impl()
        expected = np.empty((10, 2, 2))
        for i in range(10):
            expected[i] = np.full((2, 2), i)
        if device == 0:
            np.testing.assert_array_equal(a, expected)
        elif device == 1:
            np.testing.assert_array_equal(a[:nteams], expected[:nteams])
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_local_array(self, device):
        target_pragma = f"target teams num_teams(1) map(tofrom: a) map(from: nthreads) device({device})"

        @njit
        def test_impl():
            a = np.zeros((32, 10), dtype=np.int32)
            nthreads = 0
            with openmp(target_pragma):
                with openmp("parallel num_threads(32)"):
                    local_array = np.empty(10, dtype=np.int32)
                    tid = omp_get_thread_num()
                    if tid == 0:
                        nthreads = omp_get_num_threads()
                    for i in range(10):
                        local_array[i] = tid
                    for i in range(10):
                        a[tid, i] = local_array[i]
            return a, nthreads

        a, nthreads = test_impl()
        expected = np.empty((32, 10), dtype=np.int32)
        for i in range(32):
            expected[i] = [i] * 10
        if device == 0:
            self.assertEqual(nthreads, 32)
            np.testing.assert_array_equal(a, expected)
        elif device == 1:
            # CPU num_threads are capped by number of cores, which can be less
            # than the provided value.
            self.assertLessEqual(nthreads, 32)
            np.testing.assert_array_equal(a[:nthreads], expected[:nthreads])
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_teams_parallel_shared_array(self, device):
        target_pragma = f"target teams num_teams(10) map(tofrom: a) map(from: nteams, nthreads) device({device})"

        @njit
        def test_impl():
            # save data from 10 teams each of 32 threads (maximally).
            a = np.zeros((10, 32), dtype=np.int32)
            nteams = 0
            nthreads = 0

            with openmp(target_pragma):
                team_shared_array = np.empty(32, dtype=np.int32)
                team_id = omp_get_team_num()
                if team_id == 0:
                    nteams = omp_get_num_teams()
                    nthreads = omp_get_num_threads()

                with openmp("parallel num_threads(32)"):
                    thread_local_array = np.empty(10, dtype=np.int32)
                    for i in range(10):
                        thread_local_array[i] = omp_get_thread_num()

                    lasum = 0
                    for i in range(10):
                        lasum += thread_local_array[i]
                    team_shared_array[omp_get_thread_num()] = lasum / 10

                for i in range(32):
                    a[team_id, i] = team_shared_array[i]

            return a, nteams, nthreads

        r, nteams, nthreads = test_impl()
        expected = np.tile(np.arange(32), (10, 1))
        if device == 0:
            np.testing.assert_array_equal(r, expected)
        elif device == 1:
            np.testing.assert_array_equal(
                r[:nteams, :nthreads], expected[:nteams, :nthreads]
            )
        else:
            raise ValueError(f"Device {device} must be 0 or 1")

    def target_teams_loop_collapse(self, device):
        target_pragma = f"""target teams loop collapse(2)
                        device({device})
                        map(tofrom: a, b, c)"""

        @njit
        def test_impl(n):
            a = np.ones((n, n))
            b = np.ones((n, n))
            c = np.zeros((n, n))
            with openmp(target_pragma):
                for i in range(n):
                    for j in range(n):
                        c[i, j] = a[i, j] + b[i, j]
            return c

        n = 10
        c = test_impl(n)
        np.testing.assert_array_equal(c, np.full((n, n), 2))

    def target_nest_teams_nest_loop_collapse(self, device):
        target_pragma = f"""target device({device}) map(tofrom: a, b, c)"""

        @njit
        def test_impl(n):
            a = np.ones((n, n))
            b = np.ones((n, n))
            c = np.zeros((n, n))
            with openmp(target_pragma):
                with openmp("teams"):
                    with openmp("loop collapse(2)"):
                        for i in range(n):
                            for j in range(n):
                                c[i, j] = a[i, j] + b[i, j]
            return c

        n = 10
        c = test_impl(n)
        np.testing.assert_array_equal(c, np.full((n, n), 2))

    def target_teams_reduction(self, device):
        target_pragma = (
            f"""target teams device({device}) map(from: nteams) reduction(+:sum)"""
        )

        @njit
        def test_impl():
            sum = 0
            nteams = 0
            with openmp(target_pragma):
                sum += 1
                with openmp("single"):
                    nteams = omp_get_num_teams()

            return nteams, sum

        nteams, sum = test_impl()
        self.assertEqual(nteams, sum)

    def target_nest_teams_reduction(self, device):
        target_pragma = (
            f"""target device({device}) map(from: nteams) map(tofrom: sum)"""
        )

        @njit
        def test_impl():
            sum = 0
            nteams = 0
            with openmp(target_pragma):
                with openmp("teams reduction(+:sum)"):
                    sum += 1
                    with openmp("single"):
                        nteams = omp_get_num_teams()

            return nteams, sum

        nteams, sum = test_impl()
        self.assertEqual(nteams, sum)

    def target_teams_distribute_parallel_for_reduction(self, device):
        target_pragma = f"""target teams distribute parallel for device({device}) reduction(+:sum)"""

        @njit
        def test_impl():
            sum = 0
            with openmp(target_pragma):
                for _ in range(1000):
                    sum += 1

            return sum

        sum = test_impl()
        self.assertEqual(sum, 1000)

    def target_nest_teams_distribute_parallel_for_reduction(self, device):
        target_pragma = f"""target map(tofrom:sum) device({device})"""

        @njit
        def test_impl():
            sum = 0
            with openmp(target_pragma):
                with openmp("teams distribute parallel for reduction(+:sum)"):
                    for _ in range(1000):
                        sum += 1

            return sum

        sum = test_impl()
        self.assertEqual(sum, 1000)

    def target_nest_teams_nest_distribute_parallel_for_reduction(self, device):
        target_pragma = f"""target map(tofrom:sum) device({device})"""

        @njit
        def test_impl():
            sum = 0
            with openmp(target_pragma):
                with openmp("teams"):
                    with openmp("distribute parallel for reduction(+:sum)"):
                        for _ in range(1000):
                            sum += 1

            return sum

        sum = test_impl()
        self.assertEqual(sum, 1000)


for memberName in dir(TestOpenmpTarget):
    if memberName.startswith("target"):
        test_func = getattr(TestOpenmpTarget, memberName)

        def make_func_with_subtest(func):
            def func_with_subtest(self):
                for device in TestOpenmpTarget.devices:
                    with self.subTest(device=device):
                        func(self, device)

            return func_with_subtest

        setattr(
            TestOpenmpTarget,
            "test_" + test_func.__name__,
            make_func_with_subtest(test_func),
        )


class TestOpenmpPi(TestOpenmpBase):
    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_pi_loop(self):
        @njit
        def test_impl(num_steps):
            step = 1.0 / num_steps

            the_sum = 0.0
            omp_set_num_threads(4)

            with openmp("parallel"):
                with openmp("for reduction(+:the_sum) schedule(static)"):
                    for j in range(num_steps):
                        x = ((j - 1) - 0.5) * step
                        the_sum += 4.0 / (1.0 + x * x)

            pi = step * the_sum
            return pi

        r = test_impl(100000)
        np.testing.assert_almost_equal(r, 3.141632653198149)

    def test_pi_loop_combined(self):
        @njit
        def test_impl(num_steps):
            step = 1.0 / num_steps

            the_sum = 0.0
            omp_set_num_threads(4)

            with openmp("parallel for reduction(+:the_sum) schedule(static)"):
                for j in range(num_steps):
                    x = ((j - 1) - 0.5) * step
                    the_sum += 4.0 / (1.0 + x * x)

            pi = step * the_sum
            return pi

        r = test_impl(100000)
        np.testing.assert_almost_equal(r, 3.141632653198149)

    def test_pi_loop_directive(self):
        def test_impl(num_steps):
            step = 1.0 / num_steps

            the_sum = 0.0
            omp_set_num_threads(4)

            with openmp("loop reduction(+:the_sum) schedule(static)"):
                for j in range(num_steps):
                    x = ((j - 1) - 0.5) * step
                    the_sum += 4.0 / (1.0 + x * x)

            pi = step * the_sum
            return pi

        r = test_impl(100000)
        np.testing.assert_almost_equal(r, 3.141632653198149)

    # Why does this pi calculated value differ from the others?
    def test_pi_spmd(self):
        @njit
        def test_impl(num_steps):
            step = 1.0 / num_steps
            MAX_THREADS = 8
            tsum = np.zeros(MAX_THREADS)

            j = 4
            omp_set_num_threads(j)
            full_sum = 0.0

            with openmp("parallel private(tid, numthreads, local_sum, x)"):
                tid = omp_get_thread_num()
                numthreads = omp_get_num_threads()
                local_sum = 0.0

                for i in range(tid, num_steps, numthreads):
                    x = (i + 0.5) * step
                    local_sum += 4.0 / (1.0 + x * x)

                tsum[tid] = local_sum

            for k in range(j):
                full_sum += tsum[k]

            pi = step * full_sum
            return pi

        r = test_impl(1000000)
        np.testing.assert_almost_equal(r, 3.1415926535897643)

    def test_pi_task(self):
        def test_pi_comp(Nstart, Nfinish, step):
            MIN_BLK = 256
            pi_sum = 0.0
            if Nfinish - Nstart < MIN_BLK:
                for i in range(Nstart, Nfinish):
                    x = (i + 0.5) * step
                    pi_sum += 4.0 / (1.0 + x * x)
            else:
                iblk = Nfinish - Nstart
                pi_sum1 = 0.0
                pi_sum2 = 0.0
                cut = Nfinish - (iblk // 2)
                with openmp("task shared(pi_sum1)"):
                    pi_sum1 = test_pi_comp(Nstart, cut, step)
                with openmp("task shared(pi_sum2)"):
                    pi_sum2 = test_pi_comp(cut, Nfinish, step)
                with openmp("taskwait"):
                    pi_sum = pi_sum1 + pi_sum2
            return pi_sum

        @njit
        def test_pi_comp_njit(Nstart, Nfinish, step):
            MIN_BLK = 256
            pi_sum = 0.0
            if Nfinish - Nstart < MIN_BLK:
                for i in range(Nstart, Nfinish):
                    x = (i + 0.5) * step
                    pi_sum += 4.0 / (1.0 + x * x)
            else:
                iblk = Nfinish - Nstart
                pi_sum1 = 0.0
                pi_sum2 = 0.0
                cut = Nfinish - (iblk // 2)
                with openmp("task shared(pi_sum1)"):
                    pi_sum1 = test_pi_comp_njit(Nstart, cut, step)
                with openmp("task shared(pi_sum2)"):
                    pi_sum2 = test_pi_comp_njit(cut, Nfinish, step)
                with openmp("taskwait"):
                    pi_sum = pi_sum1 + pi_sum2
            return pi_sum

        def test_impl(lb, num_steps, pi_comp_func):
            step = 1.0 / num_steps

            j = 4
            omp_set_num_threads(j)
            full_sum = 0.0

            with openmp("parallel"):
                with openmp("single"):
                    full_sum = pi_comp_func(lb, num_steps, step)

            pi = step * full_sum
            return pi

        py_output = test_impl(0, 1024, test_pi_comp)
        njit_output = njit(test_impl)(0, 1024, test_pi_comp_njit)
        self.assert_outputs_equal(py_output, njit_output)


if __name__ == "__main__":
    unittest.main()
