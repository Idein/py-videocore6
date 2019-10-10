
from videocore6.driver import Driver

def test_mem():

    with Driver() as drv:

        n = 4
        a = [None] * n
        off = 42

        for i in range(n):
            a[i] = drv.alloc((256 * 1024), dtype = 'uint32')
            a[i][:] = range(i, a[i].shape[0] * n, n)
            a[i][:] += off

        for i in range(n):
            assert all(a[i][:] == range(i + off, a[i].shape[0] * n + off, n))
