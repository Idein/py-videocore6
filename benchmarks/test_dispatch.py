
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
from bench_helper import BenchHelper

@qpu
def qpu_write_N(asm, N):

    eidx(r0, sig = ldunif)
    nop(sig = ldunifrf(rf0))
    shl(r0, r0, 2)
    mov(tmud, N)
    add(tmua, r5, r0)
    tmuwt()

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

def test_multiple_dispatch_delay():
    print()

    bench = BenchHelper('benchmarks/libbench_helper.so')

    with Driver() as drv:

        data = drv.alloc((10, 16), dtype = 'uint32')
        code = [drv.program(lambda asm: qpu_write_N(asm, i)) for i in range(data.shape[0])]
        unif = drv.alloc((data.shape[0], 2), dtype = 'uint32')
        done = drv.alloc(1, dtype = 'uint32')

        data[:] = 0
        unif[:,0] = data.addresses()[:,0]
        unif[:,1] = done.addresses()[0]

        ref_start = time.time()
        with drv.compute_shader_dispatcher() as csd:
            for i in range(data.shape[0]):
                csd.dispatch(code[i], unif.addresses()[i,0])
        ref_end = time.time()
        assert (data == np.arange(data.shape[0]).reshape(data.shape[0],1)).all()

        data[:] = 0

        naive_results = np.zeros(data.shape[0], dtype='float32')
        with drv.compute_shader_dispatcher() as csd:
            for i in range(data.shape[0]):
                done[:] = 0
                start = time.time()
                csd.dispatch(code[i], unif.addresses()[i,0])
                bench.wait_address(done)
                end = time.time()
                naive_results[i] = end - start
        assert (data == np.arange(data.shape[0]).reshape(data.shape[0],1)).all()

        sleep_results = np.zeros(data.shape[0], dtype='float32')
        with drv.compute_shader_dispatcher() as csd:
            for i in range(data.shape[0]):
                done[:] = 0
                time.sleep(1)
                start = time.time()
                csd.dispatch(code[i], unif.addresses()[i,0])
                bench.wait_address(done)
                end = time.time()
                sleep_results[i] = end - start
        assert (data == np.arange(data.shape[0]).reshape(data.shape[0],1)).all()

        print
        print(f'API wait after {data.shape[0]} dispatch: {ref_end - ref_start:.6f} sec')
        print(f'polling wait for each {data.shape[0]} dispatch:')
        print(f'    total: {np.sum(naive_results):.6f} sec')
        print(f'    details: {" ".join([f"{t:.6f}" for t in naive_results])}')
        print(f'polling wait for each {data.shape[0]} dispatch with between sleep:')
        print(f'    total: {np.sum(sleep_results):.6f} sec + sleep...')
        print(f'    details: {" ".join([f"{t:.6f}" for t in sleep_results])}')
