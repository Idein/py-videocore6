
from time import monotonic

import numpy as np

from videocore6.assembler import qpu
from videocore6.driver import Driver


@qpu
def qpu_summation(asm, *, num_qpus, unroll_shift, code_offset,
                  align_cond=lambda pos: pos % 512 == 170):

    g = globals()
    for i, v in enumerate(['length', 'src', 'dst', 'qpu_num', 'stride', 'sum']):
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

    # addr += 4 * (thread_num + 16 * qpu_num)
    shl(r0, reg_qpu_num, 4)
    eidx(r1)
    add(r0, r0, r1)
    shl(r0, r0, 2)
    add(reg_src, reg_src, r0).add(reg_dst, reg_dst, r0)

    # stride = 4 * 16 * num_qpus
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 6 + num_qpus_shift)

    # The QPU performs shifts and rotates modulo 32, so it actually supports
    # shift amounts [0, 31] only with small immediates.
    num_shifts = [*range(16), *range(-16, 0)]

    # length /= 16 * 8 * num_qpus * unroll
    shr(reg_length, reg_length, num_shifts[7 + num_qpus_shift + unroll_shift])

    # This single thread switch and two instructions just before the loop are
    # really important for TMU read to achieve a better performance.
    # This also enables TMU read requests without the thread switch signal, and
    # the eight-depth TMU read request queue.
    nop(sig=thrsw)
    nop()
    bxor(reg_sum, 1, 1).mov(r1, 1)

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as l:

        unroll = 1 << unroll_shift

        for i in range(7):
            mov(tmua, reg_src).add(reg_src, reg_src, reg_stride)
        mov(tmua, reg_src).sub(reg_length, reg_length, r1, cond='pushz')
        add(reg_src, reg_src, reg_stride, sig=ldtmu(r0))

        for j in range(unroll - 1):
            for i in range(8):
                mov(tmua, reg_src).add(reg_src, reg_src, reg_stride)
                add(reg_sum, reg_sum, r0, sig=ldtmu(r0))

        for i in range(5):
            add(reg_sum, reg_sum, r0, sig=ldtmu(r0))

        l.b(cond='na0')
        add(reg_sum, reg_sum, r0, sig=ldtmu(r0))  # delay slot
        add(reg_sum, reg_sum, r0, sig=ldtmu(r0))  # delay slot
        add(reg_sum, reg_sum, r0)                 # delay slot

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


def summation(*, length, num_qpus=8, unroll_shift=5):

    assert length > 0
    assert length % (16 * 8 * num_qpus * (1 << unroll_shift)) == 0

    print(f'==== summaton example ({length / 1024 / 1024} Mi elements) ====')

    with Driver(data_area_size=(length + 1024) * 4) as drv:

        code = drv.program(qpu_summation, num_qpus=num_qpus,
                           unroll_shift=unroll_shift,
                           code_offset=drv.code_pos // 8)

        print('Preparing for buffers...')

        X = drv.alloc(length, dtype='uint32')
        Y = drv.alloc(16 * num_qpus, dtype='uint32')

        X[:] = np.arange(length, dtype=X.dtype)
        Y.fill(0)

        assert sum(Y) == 0

        unif = drv.alloc(3, dtype='uint32')
        unif[0] = length
        unif[1] = X.addresses()[0]
        unif[2] = Y.addresses()[0]

        print('Executing on QPU...')

        start = monotonic()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        end = monotonic()

        assert sum(Y) % 2**32 == (length - 1) * length // 2 % 2**32

        print(f'{end - start} sec, {length * 4 / (end - start) * 1e-6} MB/s')


def main():

    summation(length=32 * 1024 * 1024)


if __name__ == '__main__':

    main()
