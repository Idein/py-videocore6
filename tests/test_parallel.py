
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np


@qpu
def cost(asm):
    shl(r0, 8, 8)
    shl(r0, r0, 8)
    L.loop
    sub(r0, r0, 1, cond = 'pushn')
    b(R.loop, cond = 'anyna')
    nop()
    nop()
    nop()

@qpu
def qpu_serial(asm):

    nop(sig = ldunifrf(rf0))
    nop(sig = ldunifrf(rf1))
    nop(sig = ldunifrf(rf2))
    nop(sig = ldunifrf(rf3))

    eidx(r0)
    shl(r0, r0, 2)
    add(rf2, rf2, r0)
    add(rf3, rf3, r0)
    shl(r3, 4, 4)

    for i in range(16):
        mov(tmua, rf2, sig = thrsw).add(rf2, rf2, r3)
        nop()
        nop()
        nop(sig = ldtmu(r0))
        mov(tmud, r0)
        mov(tmua, rf3, sig = thrsw).add(rf3, rf3, r3)
        tmuwt()

    cost(asm)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

# This code requires 16 thread execution.
# If # of thread < 16, thread id (= (tidx & 0b111110) >> 1) could be discontiguous.
# If # of thread > 16, thread id (= (tidx & 0b111110) >> 1) could be duplicated.
@qpu
def qpu_parallel_16(asm):

    tidx(r0, sig = ldunifrf(rf0)) # rf0 = unif[0,0]
    shr(r2, r0, 2)
    band(r1, r0, 0b11)            # thread_id
    band(r2, r2, 0b1111)          # qpu_id
    shr(r1, r1, 1)
    shl(r2, r2, 1)
    add(rf31, r1, r2)             # rf31 = (qpu_id * 2) + (thread_id >> 1)

    # rf31 * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))
    nop(sig = ldunifrf(rf1))      # rf1 = unif[0,1]
    shl(r0, rf1, 2)
    umul24(r0, r0, rf31)
    add(r1, rf0, 8)
    add(r0, r0, r1)
    eidx(r1)
    shl(r1, r1, 2)
    add(tmua, r0, r1, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r0))                          # unif[th,2:18]
    mov(r5rep, r0)
    mov(rf2, r5).rotate(r5rep, r0, -1)            # rf2 = unif[th,2]
    mov(rf3, r5)                                  # rf3 = unif[th,3]

    eidx(r2)
    shl(r2, r2, 2)
    add(tmua, rf2, r2, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf32))

    eidx(r2)
    shl(r2, r2, 2)
    mov(tmud, rf32)
    add(tmua, rf3, r2)
    tmuwt()

    cost(asm)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_parallel_16():

    with Driver() as drv:

        thread = 16

        serial_code = drv.program(qpu_serial)
        parallel_code = drv.program(qpu_parallel_16)
        X = drv.alloc((thread, 16), dtype = 'int32')
        Ys = drv.alloc((thread, 16), dtype = 'int32')
        Yp = drv.alloc((thread, 16), dtype = 'int32')
        unif = drv.alloc((thread, 4), dtype = 'uint32')

        X[:] = np.random.randn(*X.shape)
        Ys[:] = -1
        Yp[:] = -1

        unif[:,0] = unif.addresses()[:,0]
        unif[:,1] = unif.shape[1]
        unif[:,2] = X.addresses()[:,0]
        unif[:,3] = Ys.addresses()[:,0]

        start = time.time()
        drv.execute(serial_code, unif.addresses()[0,0], thread=thread)
        end = time.time()
        serial_cost = end - start

        unif[:,3] = Yp.addresses()[:,0]

        start = time.time()
        drv.execute(parallel_code, unif.addresses()[0,0], thread=thread)
        end = time.time()
        parallel_cost = end - start

        np.set_printoptions(threshold=np.inf)

        assert (X == Ys).all()
        assert (X == Yp).all()
        assert parallel_cost < serial_cost * 2
