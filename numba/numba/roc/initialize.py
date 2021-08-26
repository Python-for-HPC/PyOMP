#### Additional initialization code ######
def _initialize_ufunc():
    from numba.np.ufunc import Vectorize

    def init_vectorize():
        from numba.roc.vectorizers import HsaVectorize

        return HsaVectorize

    Vectorize.target_registry.ondemand['roc'] = init_vectorize


def _initialize_gufunc():
    from numba.np.ufunc import GUVectorize

    def init_guvectorize():
        from numba.roc.vectorizers import HsaGUFuncVectorize

        return HsaGUFuncVectorize

    GUVectorize.target_registry.ondemand['roc'] = init_guvectorize


_initialize_ufunc()
_initialize_gufunc()
