
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np
import matplotlib.pyplot as plt
from bench_helper import BenchHelper


@qpu
def qpu_tmu_load_1_slot_1_qpu(asm, nops):

    nop(sig = ldunifrf(rf0)) # X.shape[1]
    nop(sig = ldunifrf(rf1)) # X
    nop(sig = ldunifrf(rf2)) # X.stride[1]
    nop(sig = ldunifrf(rf3)) # X.stride[0]
    nop(sig = ldunifrf(rf4)) # Y
    nop(sig = ldunifrf(rf5)) # done

    barrierid(syncb, sig = thrsw)
    nop()
    nop()

    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b1111, cond = 'pushz')
    b(R.done, cond = 'allna')
    nop() # delay slot
    nop() # delay slot
    nop() # delay slot

    eidx(r0)
    shl(r0, r0, 2)
    add(rf4, rf4, r0)

    eidx(r0)
    umul24(r0, r0, rf3)
    add(rf1, rf1, r0)

    mov(r2, 0.0)
    with loop as l:
        mov(tmua, rf1).add(rf1, rf1, rf2)
        for i in range(nops):
            nop()
        nop(sig = ldtmu(r3))
        sub(rf0, rf0, 1, cond = 'pushz')
        l.b(cond = 'anyna')
        fadd(r2, r2, r3) # delay slot
        nop()            # delay slot
        nop()            # delay slot

    mov(tmud, r2)
    mov(tmua, rf4)
    tmuwt()

    mov(tmud, 1)
    mov(tmua, rf5)
    tmuwt()

    L.done
    barrierid(syncb, sig = thrsw)
    nop()
    nop()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_tmu_load_1_slot_1_qpu():

    bench = BenchHelper('benchmarks/libbench_helper.so')

    for trans in [False, True]:

        with Driver() as drv:

            loop = 2**15

            X = drv.alloc((16, loop) if trans else (loop, 16), dtype = 'float32')
            Y = drv.alloc(16, dtype = 'float32')
            unif = drv.alloc(6, dtype = 'uint32')
            done = drv.alloc(1, dtype = 'uint32')

            unif[0] = loop
            unif[1] = X.addresses()[0,0]
            unif[2] = X.strides[int(trans)]
            unif[3] = X.strides[1-int(trans)]
            unif[4] = Y.addresses()[0]
            unif[5] = done.addresses()[0]

            results = np.zeros((24, 10), dtype = 'float32')

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_title(f'TMU load latency (1 slot, 1 qpu, stride=({unif[2]},{unif[3]}))')
            ax.set_xlabel('# of nop (between request and load signal)')
            ax.set_ylabel('sec')

            print()
            for nops in range(results.shape[0]):

                code = drv.program(lambda asm: qpu_tmu_load_1_slot_1_qpu(asm, nops))

                for i in range(results.shape[1]):

                    with drv.compute_shader_dispatcher() as csd:

                        X[:] = np.random.randn(*X.shape) / X.shape[int(trans)]
                        Y[:] = 0.0
                        done[:] = 0

                        start = time.time()
                        csd.dispatch(code, unif.addresses()[0], thread = 8)
                        bench.wait_address(done)
                        end = time.time()

                        results[nops,i] = end - start

                        assert np.allclose(Y, np.sum(X, axis=int(trans)), atol = 1e-4)

                ax.scatter(np.zeros(results.shape[1])+nops, results[nops], s=1, c='blue')

                print('{:4}/{}\t{:.9f}'.format(nops, results.shape[0], np.sum(results[nops]) / results.shape[1]))

            ax.set_ylim(auto=True)
            ax.set_xlim(0, results.shape[0])
            fig.savefig(f'benchmarks/tmu_load_1_slot_1_qpu_{unif[2]}_{unif[3]}.png')

@qpu
def qpu_tmu_load_2_slot_1_qpu(asm, nops):

    nop(sig = ldunifrf(rf0)) # X.shape[1]
    nop(sig = ldunifrf(rf1)) # X
    nop(sig = ldunifrf(rf2)) # X.stride[1]
    nop(sig = ldunifrf(rf3)) # X.stride[0]
    nop(sig = ldunifrf(rf4)) # Y
    nop(sig = ldunifrf(rf5)) # done

    barrierid(syncb, sig = thrsw)
    nop()
    nop()

    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b0011, cond = 'pushz')
    b(R.skip_bench, cond = 'allna')
    nop()
    nop()
    nop()

    eidx(r0)
    shl(r0, r0, 2)
    add(rf4, rf4, r0)
    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b1111)
    shl(r1, 4, 4)
    umul24(r0, r0, r1)
    add(rf4, rf4, r0)

    eidx(r0)
    umul24(r0, r0, rf3)
    add(rf1, rf1, r0)
    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b1111)
    shl(r1, rf0, 6)
    umul24(r0, r0, r1)
    add(rf1, rf1, r0)

    mov(r2, 0.0)
    with loop as l:
        mov(tmua, rf1).add(rf1, rf1, rf2)
        for i in range(nops):
            nop()
        nop(sig = ldtmu(r3))
        sub(rf0, rf0, 1, cond = 'pushz')
        l.b(cond = 'anyna')
        fadd(r2, r2, r3) # delay slot
        nop()            # delay slot
        nop()            # delay slot

    mov(tmud, r2)
    mov(tmua, rf4)
    tmuwt()

    L.skip_bench

    barrierid(syncb, sig = thrsw)
    nop()
    nop()

    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b1111, cond = 'pushz')
    b(R.skip_done, cond = 'allna')
    nop()
    nop()
    nop()
    mov(tmud, 1)
    mov(tmua, rf5)
    tmuwt()
    L.skip_done

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_tmu_load_2_slot_1_qpu():

    bench = BenchHelper('benchmarks/libbench_helper.so')

    for trans, min_nops, max_nops in [(False, 0, 64), (True, 128-32, 128+32)]:

        with Driver() as drv:

            loop = 2**13

            X = drv.alloc((8, 16, loop) if trans else (8, loop, 16), dtype = 'float32')
            Y = drv.alloc((8, 16), dtype = 'float32')
            unif = drv.alloc(6, dtype = 'uint32')
            done = drv.alloc(1, dtype = 'uint32')

            unif[0] = loop
            unif[1] = X.addresses()[0,0,0]
            unif[2] = X.strides[1+int(trans)]
            unif[3] = X.strides[2-int(trans)]
            unif[4] = Y.addresses()[0,0]
            unif[5] = done.addresses()[0]

            results = np.zeros((max_nops, 10), dtype = 'float32')

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_title(f'TMU load latency (2 slot, 1 qpu, stride=({unif[2]},{unif[3]}))')
            ax.set_xlabel('# of nop (between request and load signal)')
            ax.set_ylabel('sec')

            print()
            for nops in range(min_nops, results.shape[0]):

                code = drv.program(lambda asm: qpu_tmu_load_2_slot_1_qpu(asm, nops))

                for i in range(results.shape[1]):

                    with drv.compute_shader_dispatcher() as csd:

                        X[:] = np.random.randn(*X.shape) / X.shape[1+int(trans)]
                        Y[:] = 0.0
                        done[:] = 0

                        start = time.time()
                        csd.dispatch(code, unif.addresses()[0], thread = 8)
                        bench.wait_address(done)
                        end = time.time()

                        results[nops,i] = end - start

                        assert np.allclose(Y[0::4], np.sum(X[0::4], axis=1+int(trans)), atol = 1e-4)
                        assert (Y[1:4] == 0).all()
                        assert (Y[5:8] == 0).all()

                ax.scatter(np.zeros(results.shape[1])+nops, results[nops], s=1, c='blue')

                print('{:4}/{}\t{:.9f}'.format(nops, results.shape[0], np.sum(results[nops]) / results.shape[1]))

            ax.set_ylim(auto=True)
            ax.set_xlim(min_nops, max_nops)
            fig.savefig(f'benchmarks/tmu_load_2_slot_1_qpu_{unif[2]}_{unif[3]}.png')
