
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

    # addr += 4 * (thread_num + 16 * qpu_num)
    shl(r0, reg_qpu_num, 4)
    eidx(r1)
    add(r0, r0, r1)
    shl(r0, r0, 2)
    add(reg_src, reg_src, r0).add(reg_dst, reg_dst, r0)

    # stride = 4 * 16 * num_qpus
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 6 + num_qpus_shift)

    # length /= 16 * 8 * num_qpus * unroll
    shr(reg_length, reg_length, 7 + num_qpus_shift + unroll_shift)

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

        for i in range(8):
            mov(tmua, reg_src).add(reg_src, reg_src, reg_stride)

        for j in range(unroll - 1):
            for i in range(8):
                nop(sig=ldtmu(r0))
                mov(tmua, reg_src).add(reg_src, reg_src, reg_stride)
                mov(tmud, r0)
                mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)

        for i in range(6):
            nop(sig=ldtmu(r0))
            mov(tmud, r0)
            mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)

        nop(sig=ldtmu(r0))
        mov(tmud, r0).sub(reg_length, reg_length, 1, cond='pushz')
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)

        l.b(cond='na0')
        nop(sig=ldtmu(r0))                                    # delay slot
        mov(tmud, r0)                                         # delay slot
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)  # delay slot

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

        X = drv.alloc(length, dtype='float32')
        Y = drv.alloc(length, dtype='float32')

        X[:] = np.arange(*X.shape, dtype=X.dtype)
        Y[:] = -X

        assert not np.array_equal(X, Y)

        unif = drv.alloc(3, dtype='uint32')
        unif[0] = length
        unif[1] = X.addresses()[0]
        unif[2] = Y.addresses()[0]

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
