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
def qpu_tmu_write(asm: Assembly) -> None:
    nop(sig=ldunif)
    mov(r1, r5, sig=ldunif)

    # r2 = addr + eidx * 4
    # rf0 = eidx
    eidx(r0).mov(r2, r5)
    shl(r0, r0, 2).mov(rf0, r0)
    add(r2, r2, r0)

    with loop as l:  # noqa: E741
        # rf0: Data to be written.
        # r0: Overwritten.
        # r2: Address to write data to.

        sub(r1, r1, 1, cond="pushz").mov(tmud, rf0)
        l.b(cond="anyna")
        # rf0 += 16
        sub(rf0, rf0, -16).mov(tmua, r2)
        # r2 += 64
        shl(r0, 4, 4)
        tmuwt().add(r2, r2, r0)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_write() -> None:
    n = 4096

    with Driver(data_area_size=n * 16 * 4 + 2 * 4) as drv:
        code = drv.program(qpu_tmu_write)
        data: Array[np.uint32] = drv.alloc(n * 16, dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(2, dtype=np.uint32)

        data[:] = 0xDEADBEAF
        unif[0] = n
        unif[1] = data.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert all(data == range(n * 16))


@qpu
def qpu_tmu_vec_write(asm: Assembly, configs: list[int], vec_offset: int) -> None:
    reg_addr = rf0
    reg_n = rf1

    nop(sig=ldunifrf(reg_addr))
    nop(sig=ldunifrf(reg_n))

    with loop as l:  # noqa: E741
        assert 1 <= len(configs) <= 4
        for i, config in enumerate(configs):
            eidx(r0)
            shl(r0, r0, 0xFFFFFFF0)  # 0xfffffff0 % 32 = 16
            assert 1 <= config <= 4
            for j in range(config):
                mov(tmud, r0).add(r0, r0, 1)

            assert 0 <= vec_offset <= 3
            # addr + 4 * 4 * eidx + 4 * vec_offset
            eidx(r0)
            shl(r0, r0, 4)
            sub(r0, r0, -4 * vec_offset)
            add(tmuau if i == 0 else tmua, reg_addr, r0)

            # addr += 4 * len(configs) * 16
            shl(r0, 4, 4)
            umul24(r0, r0, len(configs))
            add(reg_addr, reg_addr, r0)

        sub(reg_n, reg_n, 1, cond="pushz")
        l.b(cond="na0")
        nop()
        nop()
        nop()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_vec_write() -> None:
    n = 123

    # The number of 32-bit values in a vector element per pixel is 1, 2, 3, or 4.
    # For example, with four 32-bit config:
    #     tmud <- r0
    #     tmud <- r1
    #     tmud <- r2
    #     tmud <- r3
    #     tmuau <- addr + 4 * 4 * eidx
    # results in:
    #     addr + 0x00: r0[ 0], r1[ 0], r2[ 0], r3[ 0], r0[ 1], r1[ 1], ..., r3[ 3]
    #     addr + 0x40: r0[ 4], r1[ 4], r2[ 4], r3[ 4], r0[ 5], r1[ 5], ..., r3[ 7]
    #     addr + 0x80: ...
    #     addr + 0xc0: r0[12], r1[12], r2[12], r3[12], r0[13], r1[13], ..., r3[15]
    # where rn[i] (0 <= i < 16) is the value in register rn of pixel (eidx) i.
    configs = [4, 3, 2, 1]

    # The element per pixel is wrapped modulo 16 bytes.
    # For example, if the above address setting is addr + 4 * 4 * eidx + 4, then
    #     addr + 0x00: r3[ 0], r0[ 0], r1[ 0], r2[ 0], r3[ 1], r0[ 1], ..., r2[ 3]
    #     addr + 0x40: r3[ 4], r0[ 4], r1[ 4], r2[ 4], r3[ 5], r0[ 5], ..., r2[ 7]
    #     addr + 0x80: ...
    #     addr + 0xc0: r3[12], r0[12], r1[12], r2[12], r3[13], r0[13], ..., r2[15]
    vec_offset = 3

    data_default = 0xDEADBEEF

    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_write, configs, vec_offset)
        data: Array[np.uint32] = drv.alloc(16 * 4 * len(configs) * n, dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(2 + n, dtype=np.uint32)

        data[:] = data_default

        unif[0] = data.addresses()[0]
        unif[1] = n

        conf = 0xFFFFFFFF
        for config in reversed(configs):
            conf <<= 8
            conf |= {1: 0xFF, 2: 0xFA, 3: 0xFB, 4: 0xFC}[config]
        conf &= 0xFFFFFFFF
        unif[2:] = conf

        drv.execute(code, unif.addresses()[0])

        for i, row in enumerate(data.reshape(-1, 4 * 16)):
            config = configs[i % len(configs)]
            for j, vec in enumerate(row.reshape(-1, 4)):
                ref = list(range(j << 16, (j << 16) + config)) + [data_default] * (4 - config)
                assert all(np.roll(vec, -vec_offset) == ref)


@qpu
def qpu_tmu_read(asm: Assembly) -> None:
    # r0: Number of vectors to read.
    # r1: Pointer to the read vectors + eidx * 4.
    # r2: Pointer to the write vectors + eidx * 4
    eidx(r2, sig=ldunif)
    mov(r0, r5, sig=ldunif)
    shl(r2, r2, 2).mov(r1, r5)
    add(r1, r1, r2, sig=ldunif)
    add(r2, r5, r2)

    with loop as l:  # noqa: E741
        mov(tmua, r1, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(rf0))

        sub(r0, r0, 1, cond="pushz").add(tmud, rf0, 1)
        l.b(cond="anyna")
        shl(r3, 4, 4).mov(tmua, r2)
        # r1 += 64
        # r2 += 64
        add(r1, r1, r3).add(r2, r2, r3)
        tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_read() -> None:
    n = 4096

    with Driver() as drv:
        code = drv.program(qpu_tmu_read)
        data: Array[np.uint32] = drv.alloc(n * 16, dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        data[:] = range(len(data))
        unif[0] = n
        unif[1] = data.addresses()[0]
        unif[2] = data.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert all(data == range(1, n * 16 + 1))


@qpu
def qpu_tmu_vec_read(asm: Assembly, configs: list[int], vec_offset: int) -> None:
    reg_src = rf0
    reg_dst = rf1
    reg_n = rf2

    nop(sig=ldunifrf(reg_src))
    nop(sig=ldunifrf(reg_dst))
    nop(sig=ldunifrf(reg_n))

    # dst += 4 * eidx
    eidx(r0)
    shl(r0, r0, 2)
    add(reg_dst, reg_dst, r0)

    with loop as l:  # noqa: E741
        mov(r4, 0)

        assert 1 <= len(configs) <= 4
        for i, config in enumerate(configs):
            assert 1 <= config <= 4
            assert 0 <= vec_offset <= 3
            # addr + 4 * 4 * eidx + 4 * vec_offset
            eidx(r0)
            shl(r0, r0, 4)
            sub(r0, r0, -4 * vec_offset)
            add(tmuau if i == 0 else tmua, reg_src, r0, sig=thrsw)
            nop()
            nop()
            nop(sig=ldtmu(r0))
            nop(sig=ldtmu(r1)) if config >= 2 else eidx(r1)
            nop(sig=ldtmu(r2)) if config >= 3 else eidx(r2)
            nop(sig=ldtmu(r3)) if config >= 4 else eidx(r3)

            add(r0, r0, r1).add(r2, r2, r3)
            add(r0, r0, r2)
            add(r4, r4, r0)
            # src += 4 * 4 * 16
            shl(r0, 4, 4)
            umul24(r0, r0, 4)
            add(reg_src, reg_src, r0)

        mov(tmud, r4)
        # If the configs are shited out, then 0xff (per-pixel regular 32-bit
        # write) is filled in.
        mov(tmua, reg_dst)

        # dst += 4 * 16
        shl(r0, 4, 4)
        add(reg_dst, reg_dst, r0)

        sub(reg_n, reg_n, 1, cond="pushz")
        l.b(cond="na0")
        nop()
        nop()
        nop()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_vec_read() -> None:
    # The settings, the number of elements in a vector, and 16-byte wrapping are
    # the same as the vector writes.

    n = 123
    configs = [4, 3, 2, 1]
    vec_offset = 1

    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_read, configs, vec_offset)
        src: Array[np.uint32] = drv.alloc((n, 16 * 4 * len(configs)), dtype=np.uint32)
        dst: Array[np.uint32] = drv.alloc((n, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3 + n, dtype=np.uint32)

        src[:, :] = np.arange(src.size, dtype=src.dtype).reshape(src.shape)
        dst[:, :] = 0

        unif[0] = src.addresses()[0, 0]
        unif[1] = dst.addresses()[0, 0]
        unif[2] = n

        conf = 0xFFFFFFFF
        for config in reversed(configs):
            conf <<= 8
            conf |= {1: 0xFF, 2: 0xFA, 3: 0xFB, 4: 0xFC}[config]
        conf &= 0xFFFFFFFF
        unif[3:] = conf

        drv.execute(code, unif.addresses()[0])

        for i, vec in enumerate(dst):
            data = src.shape[1] * i + np.arange(src.shape[1], dtype=np.uint32).reshape(len(configs), 16, 4)
            s = [0] * 16
            for j, config in enumerate(configs):
                for eidx in range(16):
                    for k in range(config):
                        s[eidx] += data[j, eidx, (k + vec_offset) % 4]
                    s[eidx] += eidx * (4 - config)
            assert all(vec == s)


# VC4 TMU cache & DMA break memory consistency.
# How about VC6 TMU ?
@qpu
def qpu_tmu_keeps_memory_consistency(asm: Assembly) -> None:
    nop(sig=ldunifrf(r0))

    mov(tmua, r0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(r1))

    add(tmud, r1, 1)
    mov(tmua, r0)
    tmuwt()

    mov(tmua, r0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(r1))

    add(tmud, r1, 1)
    mov(tmua, r0)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_keeps_memory_consistency() -> None:
    with Driver() as drv:
        code = drv.program(qpu_tmu_keeps_memory_consistency)
        data: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        data[:] = 1
        unif[0] = data.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert (data[0] == 3).all()
        assert (data[1:] == 1).all()


@qpu
def qpu_tmu_read_tmu_write_uniform_read(asm: Assembly) -> None:
    eidx(r0, sig=ldunifrf(rf0))
    shl(r0, r0, 2)
    add(rf0, rf0, r0, sig=ldunifrf(rf1))
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(r0))  # r0 = [1,...,1]

    add(tmud, r0, 1)
    mov(tmua, rf0)  # data = [2,...,2]
    tmuwt()

    b(R.set_unif_addr, cond="always").unif_addr(rf0)  # unif_addr = data.addresses()[0]
    nop()
    nop()
    nop()
    L.set_unif_addr

    nop(sig=ldunifrf(r0))  # r0 = [data[0],...,data[0]] = [2,...,2]

    add(tmud, r0, 1)
    mov(tmua, rf1)  # result = [3,...,3]
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_read_tmu_write_uniform_read() -> None:
    with Driver() as drv:
        code = drv.program(qpu_tmu_read_tmu_write_uniform_read)
        data: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)
        result: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        data[:] = 1
        unif[0] = data.addresses()[0]
        unif[1] = result.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert (data == 2).all()
        assert (result == 2).all()  # !? not 3 ?
