
from time import monotonic

import numpy as np

from videocore6.assembler import qpu
from videocore6.driver import Driver


@qpu
def qpu_memset(asm, *, num_qpus, unroll_shift, code_offset,
               align_cond=lambda pos: pos % 512 == 0):

    g = globals()
    for i, v in enumerate(['dst', 'fill', 'length', 'qpu_num', 'stride']):
        g[f'reg_{v}'] = rf[i]

    nop(sig=ldunifrf(reg_dst))
    nop(sig=ldunifrf(reg_fill))
    nop(sig=ldunifrf(reg_length))

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
    add(reg_dst, reg_dst, r0)

    # stride = 4 * 16 * num_qpus
    # r0 = 1
    mov(r0, 1)
    shl(reg_stride, r0, 6 + num_qpus_shift)

    # length /= 16 * num_qpus * unroll
    shr(reg_length, reg_length, 4 + num_qpus_shift + unroll_shift)

    unroll = 1 << unroll_shift

    if unroll == 1:

        sub(reg_length, reg_length, r0, cond='pushz')

        while not align_cond(code_offset + len(asm)):
            nop()

        with loop as l:

            l.b(cond='na0')
            mov(tmud, reg_fill)                                   # delay slot
            mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)  # delay slot
            sub(reg_length, reg_length, r0, cond='pushz')         # delay slot

    else:

        while not align_cond(code_offset + len(asm)):
            nop()

        with loop as l:

            for i in range(unroll - 2):
                mov(tmud, reg_fill)
                mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)

            mov(tmud, reg_fill).sub(reg_length, reg_length, r0, cond='pushz')
            l.b(cond='na0')
            mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)  # delay slot
            mov(tmud, reg_fill)                                   # delay slot
            mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)  # delay slot

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def memset(*, fill, length, num_qpus=8, unroll_shift=1):

    assert length > 0
    assert length % (16 * num_qpus * (1 << unroll_shift)) == 0

    print(f'==== memset example ({length * 4 / 1024 / 1024} MiB) ====')

    with Driver(data_area_size=(length + 1024) * 4) as drv:

        code = drv.program(qpu_memset, num_qpus=num_qpus,
                           unroll_shift=unroll_shift,
                           code_offset=drv.code_pos // 8)

        print('Preparing for buffers...')

        X = drv.alloc(length, dtype='uint32')

        X.fill(~fill)

        assert not np.array_equiv(X, fill)

        unif = drv.alloc(3, dtype='uint32')
        unif[0] = X.addresses()[0]
        unif[1] = fill
        unif[2] = length

        print('Executing on QPU...')

        start = monotonic()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        end = monotonic()

        assert np.array_equiv(X, fill)

        print(f'{end - start} sec, {length * 4 / (end - start) * 1e-6} MB/s')


def main():

    memset(fill=0x5a5a5a5a, length=16 * 1024 * 1024)


if __name__ == '__main__':

    main()
