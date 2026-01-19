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

import numpy as np
from bench_helper import BenchHelper

from videocore6.assembler import *
from videocore6.driver import Array, Driver


@qpu
def qpu_clock(asm: Assembly) -> None:
    nop(sig=ldunif)
    nop(sig=ldunifrf(rf0))

    with loop as l:  # noqa: E741
        sub(r5, r5, 1, cond="pushn")
        l.b(cond="anyna")
        nop()
        nop()
        nop()

    mov(tmud, 1)
    mov(tmua, rf0)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_clock() -> None:
    print()

    bench = BenchHelper("benchmarks/libbench_helper.so")

    with Driver() as drv:
        f = pow(2, 25)

        code = drv.program(qpu_clock)
        unif: Array[np.uint32] = drv.alloc(2, dtype=np.uint32)
        done: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        done[:] = 0

        unif[0] = f
        unif[1] = done.addresses()[0]

        with drv.compute_shader_dispatcher() as csd:
            start = time.time()
            csd.dispatch(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])
            bench.wait_address(done)
            end = time.time()

        print(f"{end - start:.6f} sec")
        print(f"{f * 5 / (end - start) / 1000 / 1000 * 4:.6f} MHz")
