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
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from videocore6.assembler import *
from videocore6.driver import Array, Driver


def sfu_sin(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    result: npt.NDArray[np.float32] = np.sin(x * np.pi)
    result[x < -0.5] = -1
    result[x > 0.5] = 1
    return result


ops: dict[str, Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]] = {
    # sfu regs/ops
    "recip": lambda x: np.float32(1) / x,
    "rsqrt": lambda x: np.float32(1) / np.sqrt(x).astype(np.float32),
    "exp": lambda x: 2**x,
    "log": np.log2,
    "sin": sfu_sin,
    "rsqrt2": lambda x: np.float32(1) / np.sqrt(x).astype(np.float32),
}


# SFU IO registers
@qpu
def qpu_sfu_regs(asm: Assembly, sfu_regs: list[str]) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)  # in
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig=ldtmu(r1))

    g = globals()
    for reg in sfu_regs:
        mov(g[reg], r1)
        nop()  # required ? enough ?
        mov(tmud, r4)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def boilerplate_sfu_regs(
    sfu_regs: list[str],
    domain_limitter: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]],
) -> None:
    with Driver() as drv:
        code = drv.program(qpu_sfu_regs, sfu_regs)
        x: Array[np.float32] = drv.alloc((16,), dtype=np.float32)
        y: Array[np.float32] = drv.alloc((len(sfu_regs), 16), dtype=np.float32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = domain_limitter(np.random.randn(*x.shape).astype(np.float32))
        y[:] = 0.0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        for ix, reg in enumerate(sfu_regs):
            msg = f"mov({reg}, None)"
            assert np.allclose(y[ix], ops[reg](x), rtol=1e-4), msg


def test_sfu_regs() -> None:
    boilerplate_sfu_regs(["recip", "exp", "sin"], lambda x: x)
    boilerplate_sfu_regs(["rsqrt", "log", "rsqrt2"], lambda x: x**2 + 1e-6)


# SFU ops
@qpu
def qpu_sfu_ops(asm: Assembly, sfu_ops: list[str]) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)  # in
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig=ldtmu(r1))

    g = globals()
    for op in sfu_ops:
        g[op](rf2, r1)  # ATTENTION: SFU ops requires rfN ?
        mov(tmud, rf2)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def boilerplate_sfu_ops(
    sfu_ops: list[str],
    domain_limitter: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]],
) -> None:
    with Driver() as drv:
        code = drv.program(qpu_sfu_ops, sfu_ops)
        x: Array[np.float32] = drv.alloc((16,), dtype=np.float32)
        y: Array[np.float32] = drv.alloc((len(sfu_ops), 16), dtype=np.float32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = domain_limitter(np.random.randn(*x.shape).astype(np.float32))
        y[:] = 0.0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        for ix, op in enumerate(sfu_ops):
            msg = f"{op}(None, None)"
            assert np.allclose(y[ix], ops[op](x), rtol=1e-4), msg


def test_sfu_ops() -> None:
    boilerplate_sfu_ops(["recip", "exp", "sin"], lambda x: x)
    boilerplate_sfu_ops(["rsqrt", "log", "rsqrt2"], lambda x: x**2 + 1e-6)
