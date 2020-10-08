
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np


# ldtmu
@qpu
def qpu_signal_ldtmu(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)        # start load X
    mov(r0, 1.0)                                         # r0 <- 1.0
    mov(r1, 2.0)                                         # r1 <- 2.0
    fadd(r0, r0, r0).fmul(r1, r1, r1, sig = ldtmu(rf31)) # r0 <- 2 * r0, r1 <- r1 ^ 2, rf31 <- X
    mov(tmud, rf31)
    mov(tmua, rf1)
    tmuwt().add(rf1, rf1, r3)
    mov(tmud, r0)
    mov(tmua, rf1)
    tmuwt().add(rf1, rf1, r3)
    mov(tmud, r1)
    mov(tmua, rf1)
    tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_signal_ldtmu():

    with Driver() as drv:

        code = drv.program(qpu_signal_ldtmu)
        X = drv.alloc((16, ), dtype = 'float32')
        Y = drv.alloc((3, 16), dtype = 'float32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.random.randn(*X.shape).astype('float32')
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y[0] == X).all()
        assert (Y[1] == 2).all()
        assert (Y[2] == 4).all()

# rot signal with rN source performs as a full rotate
@qpu
def qpu_full_rotate(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r0))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().add(r1, r0, r0, sig = rot(i))
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop() # require
        nop().add(r1, r0, r0, sig = rot(i))
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_full_rotate():

    with Driver() as drv:

        code = drv.program(qpu_full_rotate)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((2, len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X,X]) * 2
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[:,ix] == expected[(-rot%16):(-rot%16)+16]).all()


# rotate alias
@qpu
def qpu_rotate_alias(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r0))
    nop() # required before rotate

    for i in range(-15, 16):
        rotate(r1, r0, i)       # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        nop().rotate(r1, r0, i) # mul alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop() # require
        rotate(r1, r0, r5)       # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop() # require
        nop().rotate(r1, r0, r5) # mul alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_rotate_alias():

    with Driver() as drv:

        code = drv.program(qpu_rotate_alias)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((4, len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X,X])
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[:,ix] == expected[(-rot%16):(-rot%16)+16]).all()


# rot signal with rfN source performs as a quad rotate
@qpu
def qpu_quad_rotate(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(rf32))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().add(r1, rf32, rf32, sig = rot(i))
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop() # require
        nop().add(r1, rf32, rf32, sig = rot(r5))
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_quad_rotate():

    with Driver() as drv:

        code = drv.program(qpu_quad_rotate)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((2, len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X.reshape(4,4)]*2, axis=1)*2
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[:,ix] == expected[:,(-rot%4):(-rot%4)+4].ravel()).all()


# quad_rotate alias
@qpu
def qpu_quad_rotate_alias(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(rf32))
    nop() # required before rotate

    for i in range(-15, 16):
        quad_rotate(r1, rf32, i)       # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        nop().quad_rotate(r1, rf32, i) # mul alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop() # require
        quad_rotate(r1, rf32, r5)       # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop() # require
        nop().quad_rotate(r1, rf32, r5) # mul alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_quad_rotate_alias():

    with Driver() as drv:

        code = drv.program(qpu_quad_rotate_alias)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((4, len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X.reshape(4,4)]*2, axis=1)
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[:,ix] == expected[:,(-rot%4):(-rot%4)+4].ravel()).all()


# instruction with r5rep dst performs as a full broadcast
@qpu
def qpu_full_broadcast(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r0))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().mov(r5rep, r0, sig = [rot(ix) for ix in [i] if ix != 0] )
        mov(tmud, r5)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_full_broadcast():

    with Driver() as drv:

        code = drv.program(qpu_full_broadcast)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = X
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[ix] == expected[(-rot%16)].repeat(16)).all()


# broadcast alias
@qpu
def qpu_broadcast_alias(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r0))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().mov(broadcast, r0, sig = [rot(ix) for ix in [i] if ix != 0] )
        mov(tmud, r5)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_broadcast_alias():

    with Driver() as drv:

        code = drv.program(qpu_broadcast_alias)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = X
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[ix] == expected[(-rot%16)].repeat(16)).all()


# instruction with r5 dst performs as a quad broadcast
@qpu
def qpu_quad_broadcast(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r0))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().mov(r5, r0, sig = [rot(ix) for ix in [i] if ix != 0] )
        mov(tmud, r5)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_quad_broadcast():

    with Driver() as drv:

        code = drv.program(qpu_quad_broadcast)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X,X])
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[ix] == expected[(-rot%16):(-rot%16)+16:4].repeat(4)).all()


# instruction with r5 dst performs as a quad broadcast
@qpu
def qpu_quad_broadcast_alias(asm):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r0))
    nop() # required before rotate

    for i in range(-15, 16):
        nop().mov(quad_broadcast, r0, sig = [rot(ix) for ix in [i] if ix != 0] )
        mov(tmud, r5)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_quad_broadcast_alias():

    with Driver() as drv:

        code = drv.program(qpu_quad_broadcast_alias)
        X = drv.alloc((16, ), dtype = 'int32')
        Y = drv.alloc((len(range(-15, 16)), 16), dtype = 'int32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.concatenate([X,X])
        for ix, rot in enumerate(range(-15, 16)):
            assert (Y[ix] == expected[(-rot%16):(-rot%16)+16:4].repeat(4)).all()
