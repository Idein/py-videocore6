# Copyright (c) 2019-2020 Idein Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import time

import numpy as np

from videocore6.assembler import *
from videocore6.driver import Array, Driver


@qpu
def cost(asm: Assembly) -> None:
    shl(r0, 8, 8)
    shl(r0, r0, 8)
    with loop as l:  # noqa: E741
        sub(r0, r0, 1, cond="pushn")
        l.b(cond="anyna")
        nop()
        nop()
        nop()


@qpu
def qpu_serial(asm: Assembly) -> None:
    nop(sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    nop(sig=ldunifrf(rf2))
    nop(sig=ldunifrf(rf3))

    eidx(r0)
    shl(r0, r0, 2)
    add(rf2, rf2, r0)
    add(rf3, rf3, r0)
    shl(r3, 4, 4)

    for i in range(16):
        mov(tmua, rf2, sig=thrsw).add(rf2, rf2, r3)
        nop()
        nop()
        nop(sig=ldtmu(r0))
        mov(tmud, r0)
        mov(tmua, rf3, sig=thrsw).add(rf3, rf3, r3)
        tmuwt()

    cost(asm)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


# This code requires 16 thread execution.
# If # of thread < 16, thread id (= (tidx & 0b111110) >> 1) could be discontiguous.
# If # of thread > 16, thread id (= (tidx & 0b111110) >> 1) could be duplicated.
@qpu
def qpu_parallel_16(asm: Assembly) -> None:
    tidx(r0, sig=ldunifrf(rf0))
    shr(r0, r0, 1).mov(r1, 1)
    shl(r1, r1, 5)
    sub(r1, r1, 1)
    band(rf31, r0, r1)  # rf31 = (qpu_id * 2) + (thread_id >> 1)

    # rf31 * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))
    nop(sig=ldunifrf(rf1))  # rf1 = unif[0,1]
    shl(r0, rf1, 2)
    umul24(r0, r0, rf31)
    add(r1, rf0, 8)
    add(r0, r0, r1)
    eidx(r1)
    shl(r1, r1, 2)
    add(tmua, r0, r1, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(r0))  # unif[th,2:18]
    mov(r5rep, r0)
    mov(rf2, r5).rotate(r5rep, r0, -1)  # rf2 = unif[th,2]
    mov(rf3, r5)  # rf3 = unif[th,3]

    eidx(r2)
    shl(r2, r2, 2)
    add(tmua, rf2, r2, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf32))

    eidx(r2)
    shl(r2, r2, 2)
    mov(tmud, rf32)
    add(tmua, rf3, r2)
    tmuwt()

    cost(asm)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_parallel_16() -> None:
    with Driver() as drv:
        thread = 16

        serial_code = drv.program(qpu_serial)
        parallel_code = drv.program(qpu_parallel_16)
        x: Array[np.float32] = drv.alloc((thread, 16), dtype=np.float32)
        ys: Array[np.float32] = drv.alloc((thread, 16), dtype=np.float32)
        yp: Array[np.float32] = drv.alloc((thread, 16), dtype=np.float32)
        unif: Array[np.uint32] = drv.alloc((thread, 4), dtype=np.uint32)

        x[:] = np.random.randn(*x.shape)
        ys[:] = -1
        yp[:] = -1

        unif[:, 0] = unif.addresses()[:, 0]
        unif[:, 1] = unif.shape[1]
        unif[:, 2] = x.addresses()[:, 0]
        unif[:, 3] = ys.addresses()[:, 0]

        start = time.time()
        drv.execute(serial_code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0, 0])
        end = time.time()
        serial_cost = end - start

        unif[:, 3] = yp.addresses()[:, 0]

        start = time.time()
        drv.execute(parallel_code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0, 0], thread=thread)
        end = time.time()
        parallel_cost = end - start

        assert (x == ys).all()
        assert (x == yp).all()
        assert parallel_cost < serial_cost * 2


# If remove `barrierid` in this code, `test_barrier` will fail.
@qpu
def qpu_barrier(asm: Assembly) -> None:
    tidx(r0, sig=ldunifrf(rf0))  # rf0 = unif[0,0]
    shr(r2, r0, 2)
    band(r1, r0, 0b11)  # thread_id
    band(r2, r2, 0b1111)  # qpu_id
    shr(r1, r1, 1)
    shl(r2, r2, 1)
    add(rf31, r1, r2)  # rf31 = (qpu_id * 2) + (thread_id >> 1)

    nop(sig=ldunifrf(rf1))  # rf1 = unif[0,1]

    # rf31 * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))
    shl(r0, rf1, 2)
    umul24(r0, r0, rf31)
    add(r1, rf0, 8)
    add(r0, r0, r1)
    eidx(r1)
    shl(r1, r1, 2)
    add(tmua, r0, r1, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(r0))  # unif[th,2:18]
    mov(r5rep, r0)
    mov(rf2, r5).rotate(r5rep, r0, -1)  # rf2 = unif[th,2]
    mov(rf3, r5)  # rf3 = unif[th,3]

    eidx(r2)
    shl(r2, r2, 2)
    add(tmua, rf2, r2, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(r0))

    mov(r1, rf31)
    shl(r1, r1, 8)
    L.loop
    sub(r1, r1, 1, cond="pushn")
    b(R.loop, cond="anyna")
    nop()
    nop()
    nop()

    eidx(r2)
    shl(r2, r2, 2)
    mov(tmud, r0)
    add(tmua, rf3, r2)
    tmuwt()

    barrierid(syncb, sig=thrsw)

    add(rf32, rf31, 1)
    band(rf32, rf32, 0b1111)  # rf32 = (rf31 + 1) mod 16

    # rf32 * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))
    shl(r0, rf1, 2)
    umul24(r0, r0, rf32)
    add(r1, rf0, 8)
    add(r0, r0, r1)
    eidx(r1)
    shl(r1, r1, 2)
    add(tmua, r0, r1, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(r0))  # unif[(th+1)%16,2:18]
    mov(r5rep, r0)
    mov(rf4, r5).rotate(r5rep, r0, -1)  # rf4 = unif[(th+1)%16,2]
    mov(rf5, r5)  # rf5 = unif[(th+1)%16,3]

    eidx(r2)
    shl(r2, r2, 2)
    add(tmua, rf5, r2, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(r0))

    eidx(r2)
    shl(r2, r2, 2)
    mov(tmud, r0)
    add(tmua, rf3, r2)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_barrier() -> None:
    with Driver() as drv:
        thread = 16

        code = drv.program(qpu_barrier)
        x: Array[np.float32] = drv.alloc((thread, 16), dtype=np.float32)
        y: Array[np.float32] = drv.alloc((thread, 16), dtype=np.float32)
        unif: Array[np.uint32] = drv.alloc((thread, 4), dtype=np.uint32)

        x[:] = np.random.randn(*x.shape)
        y[:] = -1

        unif[:, 0] = unif.addresses()[:, 0]
        unif[:, 1] = unif.shape[1]
        unif[:, 2] = x.addresses()[:, 0]
        unif[:, 3] = y.addresses()[:, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0, 0], thread=thread)

        assert (y == np.concatenate([x[1:], x[:1]])).all()
