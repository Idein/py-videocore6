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

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from videocore6.assembler import *
from videocore6.driver import Array, Driver


def rotate_right(n: int, s: int) -> int:
    return ((n << (32 - s)) | (n >> s)) & 0xFFFFFFFF


def count_leading_zeros(n: int) -> int:
    bit = 0x80000000
    count = 0
    while bit != n & bit:
        count += 1
        bit >>= 1
    return count


ops: dict[str | None, Callable[..., Any]] = {
    # binary ops
    "fadd": lambda a, b: a + b,
    "faddnf": lambda a, b: a + b,
    "fsub": lambda a, b: a - b,
    "fmin": np.minimum,
    "fmax": np.maximum,
    "fmul": lambda a, b: a * b,
    "fcmp": lambda a, b: a - b,
    "vfpack": lambda a, b: np.stack([a, b]).T.ravel(),
    "vfmin": np.minimum,
    "vfmax": np.maximum,
    "vfmul": lambda a, b: a * b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "imin": np.minimum,
    "imax": np.maximum,
    "umin": np.minimum,
    "umax": np.maximum,
    "shl": lambda a, b: a << (b % 32),
    "shr": lambda a, b: a >> (b % 32),
    "asr": lambda a, b: a.astype(np.int32) >> (b % 32),
    "ror": lambda a, b: np.vectorize(rotate_right)(a, b % 32),
    "band": lambda a, b: a & b,
    "bor": lambda a, b: a | b,
    "bxor": lambda a, b: a ^ b,
    # unary ops
    "fmov": lambda x: x,
    "fround": np.round,
    "ftrunc": np.trunc,
    "ffloor": np.floor,
    "fceil": np.ceil,
    "fdx": lambda x: (x[1::2] - x[0::2]).repeat(2),
    "fdy": lambda x: (lambda a: (a[1::2] - a[0::2]).ravel())(x.reshape(-1, 2).repeat(2, axis=0).reshape(-1, 4)),
    "ftoin": lambda x: x.round().astype(np.int32),
    "ftoiz": lambda x: np.float32(x).astype(np.int32),
    "ftouz": np.vectorize(lambda x: np.float32(x).astype(np.uint32) if x > -1 else 0),
    "bnot": lambda x: ~x,
    "neg": lambda x: -x,
    "itof": lambda x: x.astype(np.float32),
    "clz": np.vectorize(count_leading_zeros),
    "utof": lambda x: x.astype(np.float32),
    # pack/unpack flags
    "l": lambda x: x[0::2],
    "h": lambda x: x[1::2],
    None: lambda x: x,
    "none": lambda x: x,
    "abs": np.abs,
    "r32": lambda x: x.repeat(2),
    "rl2h": lambda x: x[0::2].repeat(2),
    "rh2l": lambda x: x[1::2].repeat(2),
    "swap": lambda x: x.reshape(-1, 2)[:, ::-1].ravel(),
}


@qpu
def qpu_binary_ops(
    asm: Assembly,
    bin_ops: list[str],
    dst_ops: list[str | None],
    src1_ops: list[str | None],
    src2_ops: list[str | None],
) -> None:
    eidx(r0, sig=ldunif)
    mov(rf0, r5, sig=ldunif)  # in
    mov(rf1, r5, sig=ldunif)  # out
    shl(r3, 4, 4).mov(rf2, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)
    add(rf2, rf2, r0)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, r3)
    nop()
    mov(tmua, rf1, sig=thrsw).add(rf1, rf1, r3)
    nop(sig=ldtmu(r1))
    nop()
    nop(sig=ldtmu(r2))

    g = globals()
    for op, pack, unpack1, unpack2 in itertools.product(bin_ops, dst_ops, src1_ops, src2_ops):
        g[op](
            r0.pack(pack) if pack is not None else r0,
            r1.unpack(unpack1) if unpack1 is not None else r1,
            r2.unpack(unpack2) if unpack2 is not None else r2,
        )
        mov(tmud, r0)
        mov(tmua, rf2)
        tmuwt().add(rf2, rf2, r3)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def boilerplate_binary_ops(
    bin_ops: list[str],
    dst: tuple[type, list[str | None]],
    src1: tuple[type, list[str | None]],
    src2: tuple[type, list[str | None]],
    domain1: tuple[Any, Any] | None = None,
    domain2: tuple[Any, Any] | None = None,
) -> None:
    dst_dtype, dst_ops = dst
    src1_dtype, src1_ops = src1
    src2_dtype, src2_ops = src2

    with Driver() as drv:
        cases = list(itertools.product(bin_ops, dst_ops, src1_ops, src2_ops))

        code = drv.program(qpu_binary_ops, bin_ops, dst_ops, src1_ops, src2_ops)
        x1: Array[Any] = drv.alloc((16 * 4 // np.dtype(src1_dtype).itemsize,), dtype=src1_dtype)
        x2: Array[Any] = drv.alloc((16 * 4 // np.dtype(src2_dtype).itemsize,), dtype=src2_dtype)
        y: Array[Any] = drv.alloc((len(cases), 16 * 4 // np.dtype(dst_dtype).itemsize), dtype=dst_dtype)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        if domain1 is None:
            if np.dtype(src1_dtype).name.startswith("float"):
                domain1 = (-(2**7), 2**7)
            else:
                info1 = np.iinfo(src1_dtype)
                domain1 = (info1.min, info1.max - int(not np.dtype(src1_dtype).name.startswith("float")))

        if domain2 is None:
            if np.dtype(src2_dtype).name.startswith("float"):
                domain2 = (-(2**7), 2**7)
            else:
                info2 = np.iinfo(src2_dtype)
                domain2 = (info2.min, info2.max - int(not np.dtype(src2_dtype).name.startswith("float")))

        if domain1[0] == domain1[1]:
            x1[:] = domain1[0]
        elif domain1[0] < domain1[1]:
            x1[:] = np.random.uniform(domain1[0], domain1[1], x1.shape).astype(src1_dtype)
        else:
            raise ValueError("Invalid domain")

        if domain2[0] == domain2[1]:
            x2[:] = domain2[0]
        elif domain2[0] < domain2[1]:
            x2[:] = np.random.uniform(domain2[0], domain2[1], x2.shape).astype(src2_dtype)
        else:
            raise ValueError("Invalid domain")

        y[:] = 0.0

        unif[0] = x1.addresses()[0]
        unif[1] = x2.addresses()[0]
        unif[2] = y.addresses()[0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        for ix, (bin_op, dst_op, src1_op, src2_op) in enumerate(cases):
            msg = f"{bin_op}({dst_op}, {src1_op}, {src2_op})"
            if np.dtype(dst_dtype).name.startswith("float"):
                assert np.allclose(ops[dst_op](y[ix]), ops[bin_op](ops[src1_op](x1), ops[src2_op](x2)), rtol=1e-2), msg
            elif np.dtype(dst_dtype).name.startswith("int") or np.dtype(dst_dtype).name.startswith("uint"):
                assert np.all(ops[dst_op](y[ix]) == ops[bin_op](ops[src1_op](x1), ops[src2_op](x2))), msg


def test_binary_ops() -> None:
    packs: list[tuple[type, list[str | None]]] = [(np.float32, [None, "none"]), (np.float16, ["l", "h"])]
    unpacks: list[tuple[type, list[str | None]]] = [(np.float32, [None, "none", "abs"]), (np.float16, ["l", "h"])]
    for dst, src1, src2 in itertools.product(packs, unpacks, unpacks):
        boilerplate_binary_ops(
            ["fadd", "faddnf", "fsub", "fmin", "fmax", "fmul", "fcmp"],
            dst,
            src1,
            src2,
        )
    packs: list[tuple[type, list[str | None]]] = [(np.float16, [None, "none"])]
    unpacks: list[tuple[type, list[str | None]]] = [(np.float32, [None, "none"]), (np.float16, ["l", "h"])]
    for dst, src1, src2 in itertools.product(packs, unpacks, unpacks):
        boilerplate_binary_ops(
            ["vfpack"],
            dst,
            src1,
            src2,
        )
    packs: list[tuple[type, list[str | None]]] = [(np.float16, [None, "none"])]
    unpacks: list[tuple[type, list[str | None]]] = [(np.float32, ["r32"]), (np.float16, ["rl2h", "rh2l", "swap"])]
    for dst, src1, src2 in itertools.product(packs, unpacks, packs):
        boilerplate_binary_ops(
            ["vfmin", "vfmax", "vfmul"],
            dst,
            src1,
            src2,
        )

    boilerplate_binary_ops(
        ["add", "sub", "imin", "imax", "asr"],
        (np.int32, [None]),
        (np.int32, [None]),
        (np.int32, [None]),
    )
    boilerplate_binary_ops(
        ["add", "sub", "umin", "umax"],
        (np.uint32, [None]),
        (np.uint32, [None]),
        (np.uint32, [None]),
    )
    boilerplate_binary_ops(
        ["shl", "shr", "ror"],
        (np.uint32, [None]),
        (np.uint32, [None]),
        (np.uint32, [None]),
    )
    boilerplate_binary_ops(
        ["band", "bor", "bxor"],
        (np.uint32, [None]),
        (np.uint32, [None]),
        (np.uint32, [None]),
    )


@qpu
def qpu_unary_ops(
    asm: Assembly,
    uni_ops: list[str],
    dst_ops: list[str | None],
    src_ops: list[str | None],
) -> None:
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
    for op, pack, unpack in itertools.product(uni_ops, dst_ops, src_ops):
        g[op](
            r0.pack(pack) if pack is not None else r0,
            r1.unpack(unpack) if unpack is not None else r1,
        )
        mov(tmud, r0)
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


def boilerplate_unary_ops(
    uni_ops: list[str],
    dst: tuple[type, list[str | None]],
    src: tuple[type, list[str | None]],
) -> None:
    dst_dtype, dst_ops = dst
    src_dtype, src_ops = src

    with Driver() as drv:
        cases = list(itertools.product(uni_ops, dst_ops, src_ops))

        code = drv.program(qpu_unary_ops, uni_ops, dst_ops, src_ops)
        x: Array[Any] = drv.alloc((16 * 4 // np.dtype(src_dtype).itemsize,), dtype=src_dtype)
        y: Array[Any] = drv.alloc((len(cases), 16 * 4 // np.dtype(dst_dtype).itemsize), dtype=dst_dtype)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.random.uniform(-(2**15), 2**15, x.shape).astype(src_dtype)
        y[:] = 0.0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0]

        drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0])

        for ix, (uni_op, dst_op, src_op) in enumerate(cases):
            msg = f"{uni_op}({dst_op}, {src_op})"
            if np.dtype(dst_dtype).name.startswith("float"):
                assert np.allclose(ops[dst_op](y[ix]), ops[uni_op](ops[src_op](x)), rtol=1e-2), msg
            elif np.dtype(dst_dtype).name.startswith("int") or np.dtype(dst_dtype).name.startswith("uint"):
                assert np.all(ops[dst_op](y[ix]) == ops[uni_op](ops[src_op](x))), msg


def test_unary_ops() -> None:
    packs: list[tuple[type, list[str | None]]] = [(np.float32, [None, "none"]), (np.float16, ["l", "h"])]
    unpacks: list[tuple[type, list[str | None]]] = [(np.float32, [None, "none", "abs"]), (np.float16, ["l", "h"])]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ["fmov"],
            dst,
            src,
        )
    packs: list[tuple[type, list[str | None]]] = [(np.float32, [None, "none"]), (np.float16, ["l", "h"])]
    unpacks: list[tuple[type, list[str | None]]] = [(np.float32, [None, "none"]), (np.float16, ["l", "h"])]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ["fround", "ftrunc", "ffloor", "fceil", "fdx", "fdy"],
            dst,
            src,
        )
    packs: list[tuple[type, list[str | None]]] = [(np.int32, [None, "none"])]
    unpacks: list[tuple[type, list[str | None]]] = [(np.float32, [None, "none"]), (np.float16, ["l", "h"])]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ["ftoin", "ftoiz"],
            dst,
            src,
        )
    packs: list[tuple[type, list[str | None]]] = [(np.uint32, [None, "none"])]
    unpacks: list[tuple[type, list[str | None]]] = [(np.float32, [None, "none"]), (np.float16, ["l", "h"])]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ["ftouz"],
            dst,
            src,
        )
    # TODO: 'ftoc': what is the meaning of this instruction ?
    # packs = [('int32', ['none'])]
    # unpacks = [('float32', ['none']), ('float16', ['l', 'h'])]
    # for dst, src in itertools.product(packs, unpacks):
    #     boilerplate_unary_ops(
    #         ['ftoc'],
    #         dst, src,
    #     )
    boilerplate_unary_ops(
        ["bnot", "neg"],
        (np.int32, [None]),
        (np.int32, [None]),
    )
    boilerplate_unary_ops(
        ["itof"],
        (np.float32, [None]),
        (np.int32, [None]),
    )
    boilerplate_unary_ops(
        ["clz"],
        (np.uint32, [None]),
        (np.uint32, [None]),
    )
    boilerplate_unary_ops(
        ["utof"],
        (np.float32, [None]),
        (np.uint32, [None]),
    )
