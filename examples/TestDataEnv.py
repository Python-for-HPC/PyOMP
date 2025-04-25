#
#  Test individual constructs from OpenMP
#
from numba.openmp import njit
import numpy as np
from numba.openmp import openmp_context as openmp
from numba.openmp import (
    omp_get_wtime,
    omp_get_thread_num,
    omp_get_num_threads,
    omp_set_num_threads,
)


##############################################################################
@njit
def testOMP():
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

    with openmp("parallel private(x) shared(zsh) firstprivate(zfp) private(ID)"):
        ID = omp_get_thread_num()
        with openmp("single"):
            nsing = nsing + 1
            numthrds = omp_get_num_threads()
            if y != 3:
                nerr = nerr + 1
                print("Shared Default status failure y = ", y, " It should equal 3")
        with openmp("single"):
            if x == 5:
                pass
        #                 nerr = nerr+1
        #                 print("Private clause failed, variable x = original variable ",x," it should be undefined")

        # verify each thread sees the same variable vsh
        with openmp("critical"):
            zsh = zsh + ID

        # test first private
        zfp = zfp + ID
        valsfp[ID] = zfp

        # setup test to see if each thread got it's own x value
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

    print(
        nerr,
        " errors when testing parallel, private, shared, firstprivate, critical  and single",
    )

    return nerr


errors = testOMP()
