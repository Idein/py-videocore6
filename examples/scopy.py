
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


from time import monotonic

import numpy as np

from videocore6.assembler import qpu
from videocore6.driver import Driver


@qpu
def qpu_scopy(asm, *, num_qpus, unroll_shift, code_offset,
              align_cond=lambda pos: pos % 512 == 259):

    g = globals()
    for i, v in enumerate(['length', 'src', 'dst', 'qpu_num', 'stride']):
        g[f'reg_{v}'] = rf[i]

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
        raise Exception('num_qpus must be 1 or 8')

    # addr += 4 * 4 * (thread_num + 16 * qpu_num)
    shl(r0, reg_qpu_num, 4)
    eidx(r1)
    add(r0, r0, r1)
    shl(r0, r0, 4)
    add(reg_src, reg_src, r0).add(reg_dst, reg_dst, r0)

    # stride = 4 * 4 * 16 * num_qpus
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 8 + num_qpus_shift)

    num_shifts = [*range(16), *range(-16, 0)]

    # length /= 16 * 8 * num_qpus * unroll
    shr(reg_length, reg_length, num_shifts[7 + num_qpus_shift + unroll_shift])

    # This single thread switch and two nops just before the loop are really
    # important for TMU read to achieve a better performance.
    # This also enables TMU read requests without the thread switch signal, and
    # the eight-depth TMU read request queue.
    nop(sig=thrsw)
    nop()
    nop()

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as l:

        unroll = 1 << unroll_shift

        # A smaller number of instructions does not necessarily mean a faster
        # operation.  Rather, complicated TMU manipulations may perform worse
        # and even cause a hardware bug.

        mov(tmuau, reg_src).add(reg_src, reg_src, reg_stride)
        mov(tmua, reg_src).add(reg_src, reg_src, reg_stride)

        for i in range(unroll - 1):
            nop(sig=ldtmu(r0))
            mov(tmud, r0, sig=ldtmu(r0))
            mov(tmud, r0, sig=ldtmu(r0))
            mov(tmud, r0)
            nop(sig=ldtmu(r0))
            mov(tmud, r0)
            mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)
            mov(tmua, reg_src).add(reg_src, reg_src, reg_stride)
            nop(sig=ldtmu(r0))
            mov(tmud, r0, sig=ldtmu(r0))
            mov(tmud, r0, sig=ldtmu(r0))
            mov(tmud, r0)
            nop(sig=ldtmu(r0))
            mov(tmud, r0)
            mov(tmuau, reg_dst).add(reg_dst, reg_dst, reg_stride)
            mov(tmua, reg_src).add(reg_src, reg_src, reg_stride)

        if unroll == 1:
            # Prefetch the next source.
            mov(tmua, reg_src)

        nop(sig=ldtmu(r0))
        mov(tmud, r0, sig=ldtmu(r0))
        mov(tmud, r0, sig=ldtmu(r0))
        mov(tmud, r0)
        nop(sig=ldtmu(r0))
        sub(reg_length, reg_length, 1, cond='pushz').mov(tmud, r0)
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)

        if unroll == 1:
            mov(tmuc, 0xfffffffc)
        nop(sig=ldtmu(r0))
        mov(tmud, r0, sig=ldtmu(r0))
        mov(tmud, r0, sig=ldtmu(r0))

        l.b(cond='na0').unif_addr(absolute=False)
        mov(tmud, r0, sig=ldtmu(r0))
        mov(tmud, r0)
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)

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


def scopy(*, length, num_qpus=8, unroll_shift=0):

    assert length > 0
    assert length % (16 * 8 * num_qpus * (1 << unroll_shift)) == 0

    print(f'==== scopy example ({length / 1024 / 1024} Mi elements) ====')

    with Driver(data_area_size=(length * 2 + 1024) * 4) as drv:

        code = drv.program(qpu_scopy, num_qpus=num_qpus,
                           unroll_shift=unroll_shift,
                           code_offset=drv.code_pos // 8)

        print('Preparing for buffers...')

        X = drv.alloc(length, dtype='uint32')
        Y = drv.alloc(length, dtype='uint32')

        X[:] = np.arange(*X.shape, dtype=X.dtype)
        Y[:] = -X

        assert not np.array_equal(X, Y)

        unif = drv.alloc(3 + (1 << unroll_shift) + 1, dtype='uint32')
        unif[0] = length
        unif[1] = X.addresses()[0]
        unif[2] = Y.addresses()[0]
        if unroll_shift == 0:
            unif[3] = 0xfc80fcfc
        else:
            unif[3: -1] = 0xfcfcfcfc
        unif[-1] = 4 * (-len(unif) + 3) & 0xFFFFFFFF

        print('Executing on QPU...')

        start = monotonic()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        end = monotonic()

        assert np.array_equal(X, Y)

        print(f'{end - start} sec, {length * 4 / (end - start) * 1e-6} MB/s')


def main():

    scopy(length=16 * 1024 * 1024)


if __name__ == '__main__':

    main()
