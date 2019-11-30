
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
def qpu_cond_update(asm):

    eidx(r0, sig = 'ldunif')
    mov(r2, r5)
    shl(r0, r0, 2)
    add(r2, r2, r0)
    shl(r1, 4, 4)

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

    for cond_update_flag in cond_update_flags:
        eidx(r0)
        band(r0, r0, 1, cond = 'pushz')
        eidx(r0)
        sub(r0, r0, 5, cond = cond_update_flag)
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

    with Driver() as drv:

        code = drv.program(qpu_cond_update)
        data = drv.alloc((24, 16), dtype = 'uint32')
        unif = drv.alloc(1, dtype = 'uint32')

        data[:] = 0

        unif[0] = data.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.array(
            [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
             [0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1],
             [1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0],
             [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1],
             [1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0],
             [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1]],
            dtype = 'uint32'
        )

        assert (data[:12] == expected).all()
        assert (data[12:] == expected).all()

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
