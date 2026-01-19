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


# ldtmu
@qpu
def qpu_signal_ldtmu(asm: Assembly) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)  # start load X
    mov(r0, 1.0)  # r0 <- 1.0
    mov(r1, 2.0)  # r1 <- 2.0
    fadd(r0, r0, r0).fmul(r1, r1, r1, sig=ldtmu(rf31))  # r0 <- 2 * r0, r1 <- r1 ^ 2, rf31 <- X
    mov(tmud, rf31)
    mov(tmua, rf1)
    tmuwt().add(rf1, rf1, r3)
    mov(tmud, r0)
    mov(tmua, rf1)
    tmuwt().add(rf1, rf1, r3)
    mov(tmud, r1)
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


def test_signal_ldtmu() -> None:
    with Driver() as drv:
        code = drv.program(qpu_signal_ldtmu)
        x: Array[np.float32] = drv.alloc((16,), dtype=np.float32)
        y: Array[np.float32] = drv.alloc((3, 16), dtype=np.float32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.random.randn(*x.shape).astype(np.float32)
        y[:] = 0.0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        assert (y[0] == x).all()
        assert (y[1] == 2).all()
        assert (y[2] == 4).all()


# rot signal with rN source performs as a full rotate
@qpu
def qpu_full_rotate(asm: Assembly) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig=ldtmu(r0))
    nop()  # required before rotate

    for i in range(-15, 16):
        nop().add(r1, r0, r0, sig=rot(i))
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop()  # require
        nop().add(r1, r0, r0, sig=rot(i))
        mov(tmud, r1)
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


def test_full_rotate() -> None:
    with Driver() as drv:
        code = drv.program(qpu_full_rotate)
        x: Array[np.int32] = drv.alloc((16,), dtype=np.int32)
        y: Array[np.int32] = drv.alloc((2, len(range(-15, 16)), 16), dtype=np.int32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.arange(16)
        y[:] = 0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        expected = np.concatenate([x, x]) * 2
        for ix, rot in enumerate(range(-15, 16)):
            assert (y[:, ix] == expected[(-rot % 16) : (-rot % 16) + 16]).all()


# rotate alias
@qpu
def qpu_rotate_alias(asm: Assembly) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig=ldtmu(r0))
    nop()  # required before rotate

    for i in range(-15, 16):
        rotate(r1, r0, i)  # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        nop().rotate(r1, r0, i)  # mul alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop()  # require
        rotate(r1, r0, r5)  # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop()  # require
        nop().rotate(r1, r0, r5)  # mul alias
        mov(tmud, r1)
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


def test_rotate_alias() -> None:
    with Driver() as drv:
        code = drv.program(qpu_rotate_alias)
        x: Array[np.int32] = drv.alloc((16,), dtype=np.int32)
        y: Array[np.int32] = drv.alloc((4, len(range(-15, 16)), 16), dtype=np.int32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.arange(16)
        y[:] = 0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        expected = np.concatenate([x, x])
        for ix, rot in enumerate(range(-15, 16)):
            assert (y[:, ix] == expected[(-rot % 16) : (-rot % 16) + 16]).all()


# rot signal with rfN source performs as a quad rotate
@qpu
def qpu_quad_rotate(asm: Assembly) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig=ldtmu(rf32))
    nop()  # required before rotate

    for i in range(-15, 16):
        nop().add(r1, rf32, rf32, sig=rot(i))
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop()  # require
        nop().add(r1, rf32, rf32, sig=rot(r5))
        mov(tmud, r1)
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


def test_quad_rotate() -> None:
    with Driver() as drv:
        code = drv.program(qpu_quad_rotate)
        x: Array[np.int32] = drv.alloc((16,), dtype=np.int32)
        y: Array[np.int32] = drv.alloc((2, len(range(-15, 16)), 16), dtype=np.int32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.arange(16)
        y[:] = 0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        expected = np.concatenate([x.reshape(4, 4)] * 2, axis=1) * 2
        for ix, rot in enumerate(range(-15, 16)):
            assert (y[:, ix] == expected[:, (-rot % 4) : (-rot % 4) + 4].ravel()).all()


# quad_rotate alias
@qpu
def qpu_quad_rotate_alias(asm: Assembly) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig=ldtmu(rf32))
    nop()  # required before rotate

    for i in range(-15, 16):
        quad_rotate(r1, rf32, i)  # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        nop().quad_rotate(r1, rf32, i)  # mul alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop()  # require
        quad_rotate(r1, rf32, r5)  # add alias
        mov(tmud, r1)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    for i in range(-15, 16):
        mov(r5, i)
        nop()  # require
        nop().quad_rotate(r1, rf32, r5)  # mul alias
        mov(tmud, r1)
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


def test_quad_rotate_alias() -> None:
    with Driver() as drv:
        code = drv.program(qpu_quad_rotate_alias)
        x: Array[np.int32] = drv.alloc((16,), dtype=np.int32)
        y: Array[np.int32] = drv.alloc((4, len(range(-15, 16)), 16), dtype=np.int32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.arange(16)
        y[:] = 0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        expected = np.concatenate([x.reshape(4, 4)] * 2, axis=1)
        for ix, rot in enumerate(range(-15, 16)):
            assert (y[:, ix] == expected[:, (-rot % 4) : (-rot % 4) + 4].ravel()).all()


# instruction with r5rep dst performs as a full broadcast
@qpu
def qpu_full_broadcast(asm: Assembly) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig=ldtmu(r0))
    nop()  # required before rotate

    for i in range(-15, 16):
        nop().mov(r5rep, r0, sig=[rot(ix) for ix in [i] if ix != 0])
        mov(tmud, r5)
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


def test_full_broadcast() -> None:
    with Driver() as drv:
        code = drv.program(qpu_full_broadcast)
        x: Array[np.int32] = drv.alloc((16,), dtype=np.int32)
        y: Array[np.int32] = drv.alloc((len(range(-15, 16)), 16), dtype=np.int32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.arange(16)
        y[:] = 0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        expected = x
        for ix, rot in enumerate(range(-15, 16)):
            assert (y[ix] == expected[(-rot % 16)].repeat(16)).all()


# broadcast alias
@qpu
def qpu_broadcast_alias(asm: Assembly) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig=ldtmu(r0))
    nop()  # required before rotate

    for i in range(-15, 16):
        nop().mov(broadcast, r0, sig=[rot(ix) for ix in [i] if ix != 0])
        mov(tmud, r5)
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


def test_broadcast_alias() -> None:
    with Driver() as drv:
        code = drv.program(qpu_broadcast_alias)
        x: Array[np.int32] = drv.alloc((16,), dtype=np.int32)
        y: Array[np.int32] = drv.alloc((len(range(-15, 16)), 16), dtype=np.int32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.arange(16)
        y[:] = 0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        expected = x
        for ix, rot in enumerate(range(-15, 16)):
            assert (y[ix] == expected[(-rot % 16)].repeat(16)).all()


# instruction with r5 dst performs as a quad broadcast
@qpu
def qpu_quad_broadcast(asm: Assembly) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig=ldtmu(r0))
    nop()  # required before rotate

    for i in range(-15, 16):
        nop().mov(r5, r0, sig=[rot(ix) for ix in [i] if ix != 0])
        mov(tmud, r5)
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


def test_quad_broadcast() -> None:
    with Driver() as drv:
        code = drv.program(qpu_quad_broadcast)
        x: Array[np.int32] = drv.alloc((16,), dtype=np.int32)
        y: Array[np.int32] = drv.alloc((len(range(-15, 16)), 16), dtype=np.int32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.arange(16)
        y[:] = 0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        expected = np.concatenate([x, x])
        for ix, rot in enumerate(range(-15, 16)):
            assert (y[ix] == expected[(-rot % 16) : (-rot % 16) + 16 : 4].repeat(4)).all()


# instruction with r5 dst performs as a quad broadcast
@qpu
def qpu_quad_broadcast_alias(asm: Assembly) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig=ldtmu(r0))
    nop()  # required before rotate

    for i in range(-15, 16):
        nop().mov(quad_broadcast, r0, sig=[rot(ix) for ix in [i] if ix != 0])
        mov(tmud, r5)
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


def test_quad_broadcast_alias() -> None:
    with Driver() as drv:
        code = drv.program(qpu_quad_broadcast_alias)
        x: Array[np.int32] = drv.alloc((16,), dtype=np.int32)
        y: Array[np.int32] = drv.alloc((len(range(-15, 16)), 16), dtype=np.int32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.arange(16)
        y[:] = 0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        expected = np.concatenate([x, x])
        for ix, rot in enumerate(range(-15, 16)):
            assert (y[ix] == expected[(-rot % 16) : (-rot % 16) + 16 : 4].repeat(4)).all()
