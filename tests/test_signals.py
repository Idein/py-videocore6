
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
