
Supported Atomic Operations
===========================

Numba provides access to some of the atomic operations supported in HSA, in the
:class:`numba.roc.atomic` class.

Example
'''''''

The following code demonstrates the use of :class:`numba.roc.atomic.add` to
count every number in [0,32) occurred in the input array in parallel::

    from numba import roc
    import numpy as np

    @roc.jit
    def hsa_atomic_histogram(ary):
        tid = roc.get_local_id(0)
        sm = roc.shared.array(32, numba.uint32)   # declare shared library
        sm[tid] = 0                               # init values to zero
        roc.barrier(1)                            # synchronize (wait for init)
        loc = ary[tid] % 32                       # ensure we are in range
        roc.atomic.add(sm, loc, 1)                # atomic add
        roc.barrier(1)                            # synchronize
        ary[tid] = sm[tid]                        # store result inplace

    ary = np.random.randint(0, 32, size=32).astype(np.uint32)
    orig = ary.copy()

    # HSA version
    hsa_atomic_histogram[1, 32](ary)

    # Expected behavior
    gold = np.zeros_like(ary)
    for i in range(orig.size):
        gold[orig[i]] += 1

    print(ary)  # HSA kernel result
    print(gold) # for comparison
