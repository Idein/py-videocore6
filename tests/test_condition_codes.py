
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np


# `cond = 'push*'` sets the conditional flag A
@qpu
def qpu_cond_push_a(asm):

    eidx(r0, sig = 'ldunif')
    mov(r2, r5)
    shl(r0, r0, 2)
    add(r2, r2, r0)
    shl(r1, 4, 4)

    cond_pairs = [
        ('pushz', 'ifa'),
        ('pushn', 'ifna'),
        ('pushc', 'ifa'),
    ]

    for cond_push, cond_if in cond_pairs:
        eidx(r0)
        sub(r0, r0, 10, cond = cond_push)
        mov(r0, 0)
        mov(r0, 1, cond = cond_if)
        mov(tmud, r0)
        mov(tmua, r2)
        tmuwt(null).add(r2, r2, r1)
        mov(r0, 0)
        nop(null).mov(r0, 1, cond = cond_if)
        mov(tmud, r0)
        mov(tmua, r2)
        tmuwt(null).add(r2, r2, r1)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)

def test_cond_push_a():

    with Driver() as drv:

        code = drv.program(qpu_cond_push_a)
        data = drv.alloc((6, 16), dtype = 'uint32')
        unif = drv.alloc(1, dtype = 'uint32')

        data[:] = 0

        unif[0] = data.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        pushz_if_expected = np.zeros((16,), dtype = 'uint32')
        pushz_if_expected[10] = 1

        pushn_ifn_expected = np.zeros((16,), dtype = 'uint32')
        pushn_ifn_expected[10:] = 1

        pushc_if_expected = np.zeros((16,), dtype = 'uint32')
        pushc_if_expected[:10] = 1

        assert (data[0] == pushz_if_expected).all()
        assert (data[1] == pushz_if_expected).all()
        assert (data[2] == pushn_ifn_expected).all()
        assert (data[3] == pushn_ifn_expected).all()
        assert (data[4] == pushc_if_expected).all()
        assert (data[5] == pushc_if_expected).all()

# `cond = 'push*'` moves the old conditional flag A to B
@qpu
def qpu_cond_push_b(asm):

    eidx(r0, sig = 'ldunif')
    mov(r2, r5)
    shl(r0, r0, 2)
    add(r2, r2, r0)
    shl(r1, 4, 4)

    eidx(r0)
    sub(null, r0, 10, cond = 'pushz')
    mov(r0, 0, cond = 'ifa')
    eidx(r0).mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    eidx(r0)
    sub(null, r0, 5, cond = 'pushz')
    mov(r0, 0, cond = 'ifa')
    eidx(r0).mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    mov(r0, 0, cond = 'ifb')
    eidx(r0).mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    eidx(r0)
    sub(null, r0, 1, cond = 'pushz')
    mov(r0, 0, cond = 'ifa')
    eidx(r0).mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    mov(r0, 0, cond = 'ifb')
    eidx(r0).mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)

def test_cond_push_b():

    with Driver() as drv:

        code = drv.program(qpu_cond_push_b)
        data = drv.alloc((5, 16), dtype = 'uint32')
        unif = drv.alloc(1, dtype = 'uint32')

        data[:] = 0

        unif[0] = data.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        push0 = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,11,12,13,14,15]
        push1 = [ 0, 1, 2, 3, 4, 0, 6, 7, 8, 9,10,11,12,13,14,15]
        push2 = [ 0, 0, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15]

        expected = np.array(
            #  pushz
            [push0,  # ifa
             # pushz
             push1,  # ifa
             push0,  # ifb
             # pushz
             push2,  # ifa
             push1], # ifb
            dtype = 'uint32'
        )

        assert (data == expected).all()

# `cond = '{and,nor}*'` updates the conditional flag A and it don't affect to B
@qpu
def qpu_cond_update(asm, cond_update_flags):

    eidx(r0, sig = 'ldunif')
    mov(r2, r5)
    shl(r0, r0, 2)
    add(r2, r2, r0)
    shl(r1, 4, 4)

    for cond_update_flag in cond_update_flags:
        eidx(r0)
        band(r0, r0, 1, cond = 'pushz') # fla = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
        eidx(r0)
        sub(null, r0, 5, cond = cond_update_flag)
        mov(r0, 0)
        mov(r0, 1, cond = 'ifa')
        mov(tmud, r0)
        mov(tmua, r2)
        tmuwt(null).add(r2, r2, r1)

    for cond_update_flag in cond_update_flags:
        eidx(r0)
        band(r0, r0, 1, cond = 'pushz')
        eidx(r0)
        add(r3, r0, r0).sub(r0, r0, 5, cond = cond_update_flag)
        mov(r0, 0)
        mov(r0, 1, cond = 'ifa')
        mov(tmud, r0)
        mov(tmua, r2)
        tmuwt(null).add(r2, r2, r1)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)

def test_cond_update():

    cond_update_flags = [
        'andz',
        'andnz',
        'nornz',
        'norz',
        'andn',
        'andnn',
        'nornn',
        'norn',
        'andc',
        'andnc',
        'nornc',
        'norc',
    ]

    def cond_update_op(cond_update_flag):
        bin_op = [
            lambda a,b: np.logical_not(np.logical_or(a, b)),
            np.logical_and
        ][cond_update_flag[:3] == 'and']
        b_op = lambda b: [b < 0, b == 0][cond_update_flag[-1] == 'z']
        not_op = [lambda x: x, np.logical_not][cond_update_flag[3:-1] == 'n']
        return lambda a,b: bin_op(a, not_op(b_op(b)))

    with Driver() as drv:

        code = drv.program(lambda asm: qpu_cond_update(asm, cond_update_flags))
        data = drv.alloc((24, 16), dtype = 'uint32')
        unif = drv.alloc(1, dtype = 'uint32')

        data[:] = 0

        unif[0] = data.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        a = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]) > 0
        b = np.arange(16) - 5

        for ix, cond_update_flag in enumerate(cond_update_flags):
            assert np.all(data[ix] == cond_update_op(cond_update_flag)(a, b))

# dual `cond=''` instruction
@qpu
def qpu_cond_combination(asm):

    eidx(r0, sig = 'ldunif')
    mov(r2, r5)
    shl(r0, r0, 2)
    add(r2, r2, r0)
    shl(r1, 4, 4)

    # if / push
    eidx(r0)
    sub(r0, r0, 10, cond = 'pushz')
    eidx(r0)
    mov(r0, 5, cond = 'ifa').sub(r3, r0, 5, cond = 'pushn')
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    eidx(r0)
    mov(r0, 0, cond = 'ifa')
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    # push / if
    eidx(r0)
    sub(r0, r0, 10, cond = 'pushz')
    eidx(r0)
    sub(null, r0, 5, cond = 'pushn').mov(r0, 5, cond = 'ifa')
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    eidx(r0)
    mov(r0, 0, cond = 'ifa')
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    # if / if
    eidx(r0)
    sub(null, r0, 10, cond = 'pushn')
    eidx(r3)
    mov(r0, 0, cond = 'ifna').mov(r3, 0, cond = 'ifna')
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    mov(tmud, r3)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    # update / if
    eidx(r0)
    sub(null, r0, 10, cond = 'pushn')
    eidx(r3)
    sub(null, r0, 5, cond = 'andn').mov(r3, 5, cond = 'ifa')
    eidx(r0)
    mov(r0, 0, cond = 'ifa')
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    mov(tmud, r3)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)

def test_cond_combination():

    with Driver() as drv:

        code = drv.program(qpu_cond_combination)
        data = drv.alloc((8, 16), dtype = 'uint32')
        unif = drv.alloc(1, dtype = 'uint32')

        data[:] = 0

        unif[0] = data.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.array(
            [[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5,11,12,13,14,15],
             [ 0, 0, 0, 0, 0, 5, 6, 7, 8, 9,10,11,12,13,14,15],
             [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5,11,12,13,14,15],
             [ 0, 0, 0, 0, 0, 5, 6, 7, 8, 9,10,11,12,13,14,15],
             [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0],
             [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 5, 6, 7, 8, 9,10,11,12,13,14,15],
             [ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,10,11,12,13,14,15]],
            dtype = 'uint32'
        )

        assert (data == expected).all()


# vflx instructions read a condition flag as int16
@qpu
def qpu_cond_vflx(asm, ops):

    eidx(r0, sig = 'ldunif')
    mov(r2, r5)
    shl(r0, r0, 2)
    add(r2, r2, r0)
    shl(r1, 4, 4)

    # init fla/flb
    bxor(rf0, rf0, rf0).sub(rf1, rf1, rf1)
    eidx(r0)
    band(null, r0, 1 << 0, cond = 'pushz') # a = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
    band(null, r0, 1 << 1, cond = 'pushz') # a = [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0], b = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]

    # flapush
    g = globals()
    for op in ops:
        g[op](r0)
        mov(tmud, r0)
        mov(tmua, r2)
        tmuwt(null).add(r2, r2, r1)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)

def test_cond_vflx():

    def expected(op):
        result = [
            np.array([1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0], dtype = 'int16'),
            np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0], dtype = 'int16'),
        ][op[-1] == 'b'].repeat(2)
        if op[3:-1] == 'n':
            result = 1 - result
        return result

    ops = [
        'vfla',
        'vflna',
        'vflb',
        'vflnb',
    ]

    with Driver() as drv:

        code = drv.program(lambda asm: qpu_cond_vflx(asm, ops))
        data = drv.alloc((len(ops), 32), dtype = 'int16')
        unif = drv.alloc(1, dtype = 'uint32')

        data[:] = 0

        unif[0] = data.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        for ix, op in enumerate(ops):
            assert (data[ix] == expected(op)).all()
