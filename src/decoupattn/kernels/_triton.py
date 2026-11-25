try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


def jit(fn):
    return triton.jit(fn) if triton is not None else fn
