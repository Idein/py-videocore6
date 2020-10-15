
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
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np


@qpu
def qpu_tmu_write(asm):

    nop(sig = ldunif)
    mov(r1, r5, sig = ldunif)

    # r2 = addr + eidx * 4
    # rf0 = eidx
    eidx(r0).mov(r2, r5)
    shl(r0, r0, 2).mov(rf0, r0)
    add(r2, r2, r0)

    with loop as l:

        # rf0: Data to be written.
        # r0: Overwritten.
        # r2: Address to write data to.

        sub(r1, r1, 1, cond = 'pushz').mov(tmud, rf0)
        l.b(cond = 'anyna')
        # rf0 += 16
        sub(rf0, rf0, -16).mov(tmua, r2)
        # r2 += 64
        shl(r0, 4, 4)
        tmuwt().add(r2, r2, r0)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()


def test_tmu_write():

    n = 4096

    with Driver(data_area_size = n * 16 * 4 + 2 * 4) as drv:

        code = drv.program(qpu_tmu_write)
        data = drv.alloc(n * 16, dtype = 'uint32')
        unif = drv.alloc(2, dtype = 'uint32')

        data[:] = 0xdeadbeaf
        unif[0] = n
        unif[1] = data.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert all(data == range(n * 16))


@qpu
def qpu_tmu_read(asm):

    # r0: Number of vectors to read.
    # r1: Pointer to the read vectors + eidx * 4.
    # r2: Pointer to the write vectors + eidx * 4
    eidx(r2, sig = ldunif)
    mov(r0, r5, sig = ldunif)
    shl(r2, r2, 2).mov(r1, r5)
    add(r1, r1, r2, sig = ldunif)
    add(r2, r5, r2)

    with loop as l:

        mov(tmua, r1, sig = thrsw)
        nop()
        nop()
        nop(sig = ldtmu(rf0))

        sub(r0, r0, 1, cond = 'pushz').add(tmud, rf0, 1)
        l.b(cond = 'anyna')
        shl(r3, 4, 4).mov(tmua, r2)
        # r1 += 64
        # r2 += 64
        add(r1, r1, r3).add(r2, r2, r3)
        tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()


def test_tmu_read():

    n = 4096

    with Driver() as drv:

        code = drv.program(qpu_tmu_read)
        data = drv.alloc(n * 16, dtype = 'uint32')
        unif = drv.alloc(3, dtype = 'uint32')

        data[:] = range(len(data))
        unif[0] = n
        unif[1] = data.addresses()[0]
        unif[2] = data.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert all(data == range(1, n * 16 + 1))


# VC4 TMU cache & DMA break memory consistency.
# How about VC6 TMU ?
@qpu
def qpu_tmu_keeps_memory_consistency(asm):

    nop(sig = ldunifrf(r0))

    mov(tmua, r0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r1))

    add(tmud, r1, 1)
    mov(tmua, r0)
    tmuwt()

    mov(tmua, r0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r1))

    add(tmud, r1, 1)
    mov(tmua, r0)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_tmu_keeps_memory_consistency():

    with Driver() as drv:

        code = drv.program(qpu_tmu_keeps_memory_consistency)
        data = drv.alloc(16, dtype = 'uint32')
        unif = drv.alloc(3, dtype = 'uint32')

        data[:] = 1
        unif[0] = data.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert (data[0] == 3).all()
        assert (data[1:] == 1).all()


@qpu
def qpu_tmu_read_tmu_write_uniform_read(asm):

    eidx(r0, sig = ldunifrf(rf0))
    shl(r0, r0, 2)
    add(rf0, rf0, r0, sig = ldunifrf(rf1))
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r0)) # r0 = [1,...,1]

    add(tmud, r0, 1)
    mov(tmua, rf0)       # data = [2,...,2]
    tmuwt()

    b(R.set_unif_addr, cond = 'always').unif_addr(rf0) # unif_addr = data.addresses()[0]
    nop()
    nop()
    nop()
    L.set_unif_addr

    nop(sig = ldunifrf(r0)) # r0 = [data[0],...,data[0]] = [2,...,2]

    add(tmud, r0, 1)
    mov(tmua, rf1)          # result = [3,...,3]
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_tmu_read_tmu_write_uniform_read():

    with Driver() as drv:

        code = drv.program(qpu_tmu_read_tmu_write_uniform_read)
        data = drv.alloc(16, dtype = 'uint32')
        result = drv.alloc(16, dtype = 'uint32')
        unif = drv.alloc(3, dtype = 'uint32')

        data[:] = 1
        unif[0] = data.addresses()[0]
        unif[1] = result.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert (data == 2).all()
        assert (result == 2).all() # !? not 3 ?
