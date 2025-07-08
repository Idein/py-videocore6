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
from collections.abc import Callable
from time import monotonic

import numpy as np

from videocore6.assembler import *
from videocore6.driver import Array, Driver


@qpu
def qpu_summation(
    asm: Assembly,
    *,
    num_qpus: int,
    unroll_shift: int,
    code_offset: int,
    align_cond: Callable[[int], bool] = lambda pos: pos % 512 == 170,
) -> None:
    reg_length = rf0
    reg_src = rf1
    reg_dst = rf2
    reg_qpu_num = rf3
    reg_stride = rf4
    reg_sum = rf5

    nop(sig=ldunifrf(reg_length))
    nop(sig=ldunifrf(reg_src))
    nop(sig=ldunifrf(reg_dst))

    if num_qpus == 1:
        num_qpus_shift = 0
        mov(reg_qpu_num, 0)
    elif num_qpus == 8:
        num_qpus_shift = 3
        tidx(r0)
        shr(r0, r0, 2)
        band(reg_qpu_num, r0, 0b1111)
    else:
        raise Exception("num_qpus must be 1 or 8")

    # src += 4 * 4 * (thread_num + 16 * qpu_num)
    # dst += 4 * (thread_num + 16 * qpu_num)
    shl(r0, reg_qpu_num, 4)
    eidx(r1)
    add(r0, r0, r1)
    shl(r0, r0, 2)
    shl(r0, r0, 2).add(reg_dst, reg_dst, r0)
    add(reg_src, reg_src, r0)

    # stride = 4 * 4 * 16 * num_qpus
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 8 + num_qpus_shift)

    # The QPU performs shifts and rotates modulo 32, so it actually supports
    # shift amounts [0, 31] only with small immediates.
    num_shifts = [*range(16), *range(-16, 0)]

    # length /= 16 * 8 * num_qpus * unroll
    shr(reg_length, reg_length, num_shifts[7 + num_qpus_shift + unroll_shift])

    # sum = 0
    # length -= 1
    # r2 = stride

    # This single thread switch and two instructions just before the loop are
    # really important for TMU read to achieve a better performance.
    # This also enables TMU read requests without the thread switch signal, and
    # the eight-depth TMU read request queue.
    nop(sig=thrsw)
    bxor(reg_sum, 1, 1).mov(r1, 1)
    sub(reg_length, reg_length, r1, cond="pushz").mov(r2, reg_stride)

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as l:  # noqa: E741
        unroll = 1 << unroll_shift

        mov(tmuau, reg_src).add(reg_src, reg_src, reg_stride)
        mov(tmua, reg_src, sig=ldtmu(r0))

        for i in range(unroll - 1):
            add(reg_sum, reg_sum, r0, sig=ldtmu(r0)).add(reg_src, reg_src, r2)
            add(reg_sum, reg_sum, r0, sig=ldtmu(r0))
            add(reg_sum, reg_sum, r0, sig=ldtmu(r0))
            add(reg_sum, reg_sum, r0, sig=ldtmu(r0)).mov(tmuau if i % 2 == 1 else tmua, reg_src)
            add(reg_sum, reg_sum, r0, sig=ldtmu(r0)).add(reg_src, reg_src, r2)
            add(reg_sum, reg_sum, r0, sig=ldtmu(r0))
            add(reg_sum, reg_sum, r0, sig=ldtmu(r0))
            add(reg_sum, reg_sum, r0, sig=ldtmu(r0)).mov(tmua, reg_src)

        add(reg_sum, reg_sum, r0, sig=ldtmu(r0)).add(reg_src, reg_src, r2)
        add(reg_sum, reg_sum, r0, sig=ldtmu(r0))
        add(reg_sum, reg_sum, r0, sig=ldtmu(r0))
        add(reg_sum, reg_sum, r0, sig=ldtmu(r0))
        add(reg_sum, reg_sum, r0, sig=ldtmu(r0))

        l.b(cond="na0").unif_addr(absolute=False)
        add(reg_sum, reg_sum, r0, sig=ldtmu(r0))
        add(reg_sum, reg_sum, r0, sig=ldtmu(r0))
        add(reg_sum, reg_sum, r0).sub(reg_length, reg_length, r1, cond="pushz")

    mov(tmud, reg_sum)
    mov(tmua, reg_dst)

    # This synchronization is needed between the last TMU operation and the
    # program end with the thread switch just before the loop above.
    barrierid(syncb, sig=thrsw)
    nop()
    nop()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def summation(*, length: int, num_qpus: int = 8, unroll_shift: int = 2) -> None:
    assert length > 0
    assert length % (16 * 8 * num_qpus * (1 << unroll_shift)) == 0

    print(f"==== summaton example ({length / 1024 / 1024} Mi elements) ====")

    with Driver(data_area_size=(length + 1024) * 4) as drv:
        code = drv.program(qpu_summation, num_qpus=num_qpus, unroll_shift=unroll_shift, code_offset=drv.code_pos // 8)

        print("Preparing for buffers...")

        x: Array[np.uint32] = drv.alloc(length, dtype=np.uint32)
        y: Array[np.uint32] = drv.alloc(16 * num_qpus, dtype=np.uint32)

        x[:] = np.arange(length, dtype=x.dtype)
        y.fill(0)

        assert sum(y) == 0

        if unroll_shift == 0:
            unif: Array[np.uint32] = drv.alloc(3 + 1 + 1, dtype=np.uint32)
            unif[3] = 0xFFFFFCFC
        else:
            unif = drv.alloc(3 + (1 << (unroll_shift - 1)) + 1, dtype=np.uint32)
            unif[3:-1] = 0xFCFCFCFC
        unif[0] = length
        unif[1] = x.addresses()[0]
        unif[2] = y.addresses()[0]
        unif[-1] = 4 * (-len(unif) + 3) & 0xFFFFFFFF

        print("Executing on QPU...")

        start = monotonic()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        end = monotonic()

        assert int(sum(y.astype(int))) % 2**32 == (length - 1) * length // 2 % 2**32

        print(f"{end - start} sec, {length * 4 / (end - start) * 1e-6} MB/s")


def main() -> None:
    summation(length=32 * 1024 * 1024)


if __name__ == "__main__":
    main()
