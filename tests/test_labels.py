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
import numpy as np

from videocore6.assembler import *
from videocore6.driver import Array, Driver


@qpu
def qpu_label_with_namespace(asm: Assembly) -> None:
    mov(r0, 0)

    with namespace("ns1"):
        b(R.test, cond="always")
        nop()
        nop()
        nop()
        add(r0, r0, 10)
        L.test
        add(r0, r0, 1)

        with namespace("nested"):
            b(R.test, cond="always")
            nop()
            nop()
            nop()
            add(r0, r0, 10)
            L.test
            add(r0, r0, 1)

    with namespace("ns2"):
        b(R.test, cond="always")
        nop()
        nop()
        nop()
        add(r0, r0, 10)
        L.test
        add(r0, r0, 1)

    b(R.test, cond="always")
    nop()
    nop()
    nop()
    add(r0, r0, 10)
    L.test
    add(r0, r0, 1)

    with namespace("ns3"):
        b(R.test, cond="always")
        nop()
        nop()
        nop()
        add(r0, r0, 10)
        L.test
        add(r0, r0, 1)

    eidx(r1, sig=ldunifrf(rf2))
    shl(r1, r1, 2)

    mov(tmud, r0)
    add(tmua, rf2, r1)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_label_with_namespace() -> None:
    with Driver() as drv:
        code = drv.program(qpu_label_with_namespace)
        data: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        data[:] = 1234

        unif[0] = data.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert (data == 5).all()
