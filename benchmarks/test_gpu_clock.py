
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
from bench_helper import BenchHelper

@qpu
def qpu_clock(asm):

    nop(sig = ldunif)
    nop(sig = ldunifrf(rf0))

    with loop as l:
        sub(r5, r5, 1, cond = 'pushn')
        l.b(cond = 'anyna')
        nop()
        nop()
        nop()

    mov(tmud, 1)
    mov(tmua, rf0)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()


def test_clock():
    print()

    bench = BenchHelper('benchmarks/libbench_helper.so')

    with Driver() as drv:

        f = pow(2, 25)

        code = drv.program(qpu_clock)
        unif = drv.alloc(2, dtype = 'uint32')
        done = drv.alloc(1, dtype = 'uint32')

        done[:] = 0

        unif[0] = f
        unif[1] = done.addresses()[0]

        with drv.compute_shader_dispatcher() as csd:
            start = time.time()
            csd.dispatch(code, unif.addresses()[0])
            bench.wait_address(done)
            end = time.time()

        print(f'{end - start:.6f} sec')
        print(f'{f * 5 / (end - start) / 1000 / 1000 * 4:.6f} MHz')
