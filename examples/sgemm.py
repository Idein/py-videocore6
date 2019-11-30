#!/usr/bin/env python3


import struct
from time import clock_gettime, CLOCK_MONOTONIC
import numpy as np
from videocore6.driver import Driver
from videocore6.assembler import qpu


def pack_unpack(pack, unpack, *args):
    return [struct.unpack(unpack, struct.pack(pack, _))[0] for _ in args]


def getsec():
    return clock_gettime(CLOCK_MONOTONIC)


@qpu
def qpu_sgemm_rnn_naive(asm):

    g = globals()

    for i, reg in enumerate(['A_i', 'A_k', 'B_i', 'B_j', 'B_k', 'C_j',
            'alpha', 'beta', 'P', 'Q', 'R', 'i', 'j', 'k', '64', '4_Q', '4_R']):
        g['reg_' + reg] = g['rf' + str(i)]

    for reg in ['A_i', 'B_i', 'C_j', 'alpha', 'beta', 'P', 'Q', 'R']:
        nop(null, sig = 'ldunif')
        mov(g['reg_' + reg], r5)

    # B_i += 4 * eidx
    # C_j += 4 * eidx
    eidx(r0)
    shl(r0, r0, 2)
    add(reg_B_i, reg_B_i, r0)
    add(reg_C_j, reg_C_j, r0)

    shl(reg_64, 4, 4)
    shl(reg_4_Q, reg_Q, 2)
    shl(reg_4_R, reg_R, 2)

    mov(reg_i, reg_P)

    L.loop_i
    if True:

        shr(reg_j, reg_R, 4)
        mov(reg_B_j, reg_B_i)

        L.loop_j
        if True:

            mov(reg_A_k, reg_A_i).mov(reg_B_k, reg_B_j)
            mov(r0, 0.).mov(reg_k, reg_Q)

            L.loop_k
            if True:

                mov(tmua, reg_A_k, sig = 'thrsw')
                sub(reg_k, reg_k, 1, cond = 'pushz')
                mov(tmua, reg_B_k, sig = 'thrsw')
                nop(r1, sig = 'ldtmu')
                nop(r2, sig = 'ldtmu')

                b(R.loop_k, cond = 'anyna')
                nop(null).fmul21(r1, r1, r2)
                fadd5(r0, r0, r1).add(reg_A_k, reg_A_k, 4)
                add(reg_B_k, reg_B_k, reg_4_R)

            mov(tmua, reg_C_j, sig = 'thrsw')
            nop(null).fmul21(r0, r0, reg_alpha)
            sub(reg_j, reg_j, 1, cond = 'pushz')
            nop(r1, sig = 'ldtmu')
            nop(null).fmul21(r1, r1, reg_beta)

            b(R.loop_j, cond = 'anyna')
            fadd5(tmud, r0, r1).add(reg_B_j, reg_B_j, reg_64)
            add(reg_C_j, reg_C_j, reg_64).mov(tmua, reg_C_j)
            tmuwt(null)

        sub(reg_i, reg_i, 1, cond = 'pushz')
        b(R.loop_i, cond = 'anyna')
        add(reg_A_i, reg_A_i, reg_4_Q)
        nop(null)
        nop(null)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)


def sgemm_rnn_naive():

    P = 123
    Q = 567
    R = 16 * 32

    assert R % 16 == 0

    with Driver() as drv:

        #drv.dump_program(qpu_sgemm_rnn_naive); exit(0)

        code = drv.program(qpu_sgemm_rnn_naive)
        unif = drv.alloc(1024, dtype = 'uint32')
        A = drv.alloc((P, Q), dtype = 'float32')
        B = drv.alloc((Q, R), dtype = 'float32')
        C = drv.alloc((P, R), dtype = 'float32')

        np.random.seed(0)
        alpha = np.random.randn()
        beta = np.random.randn()
        A[:] = np.random.randn(P, Q)
        B[:] = np.random.randn(Q, R)
        C[:] = np.random.randn(P, R)

        start = getsec()
        C_ref = alpha * A.dot(B) + beta * C
        time_ref = getsec() - start

        for i, x in enumerate([A.addresses()[0, 0], B.addresses()[0, 0],
                C.addresses()[0, 0], *pack_unpack('f', 'I', alpha, beta),
                P, Q, R]):
            unif[i] = x

        start = getsec()
        drv.execute(code, unif.addresses()[0])
        time_gpu = getsec() - start

        def Gflops(sec):
            return (2 * P * Q * R + 3 * P * R) / sec * 1e-9

        print(f'==== sgemm example ({P}x{Q} times {Q}x{R}) ====')
        print(f'numpy: {time_ref:.4} sec, {Gflops(time_ref):.4} Gflop/s')
        print(f'QPU:   {time_gpu:.4} sec, {Gflops(time_gpu):.4} Gflop/s')
        print(f'Minimum absolute error: {np.min(np.abs(C - C_ref))}')
        print(f'Maximum absolute error: {np.max(np.abs(C - C_ref))}')
        print(f'Minimum relative error: {np.min(np.abs((C - C_ref) / C_ref))}')
        print(f'Maximum relative error: {np.max(np.abs((C - C_ref) / C_ref))}')


def main():

    sgemm_rnn_naive()


if __name__ == '__main__':
    main()
