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


@qpu
def qpu_tmu_vec_n_naive_write(asm: Assembly, vec_n: int) -> None:
    assert 2 <= vec_n <= 4

    eidx(rf1, cond="pushz")
    nop(sig=ldunifrf(rf2))
    mov(rf10, rf2, cond="ifa")
    for i in range(15):
        nop(sig=ldunifrf(rf2))
        sub(rf1, rf1, 1, cond="pushz")
        mov(rf10, rf2, cond="ifa")

    mov(rf1, 8)
    add(rf1, rf1, 8)
    eidx(rf20)
    add(rf20, rf20, 1)
    add(rf21, rf20, rf1)
    add(rf22, rf21, rf1)
    add(rf23, rf22, rf1)

    bnot(tmuc, 7 - vec_n)  # vec n write
    for i in range(vec_n):
        mov(tmud, rf[20 + i])
    mov(tmua, rf10)

    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_vec4_naive_write() -> None:
    # result[:] <- 0 (result.addresses()[0, 0] must be aligned to 64 bytes)
    #
    # tmuc <- vec4 write
    #
    # tmud[0] <- [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
    # tmud[1] <- [17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]
    # tmud[2] <- [33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48]
    # tmud[3] <- [49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64]
    #
    # tmua <- result.addresses()[i, i]
    #
    # result =
    # [[ 1 17 33 49  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [50  2 18 34  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [35 51  3 19  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [20 36 52  4  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  5 21 37 53  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0 54  6 22 38  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0 39 55  7 23  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0 24 40 56  8  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  9 25 41 57  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0 58 10 26 42  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0 43 59 11 27  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0 28 44 60 12  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 13 29 45 61]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 62 14 30 46]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 47 63 15 31]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 32 48 64 16]]
    #
    vec = 4
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_naive_write, vec)
        result: Array[np.uint32] = drv.alloc((16, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)

        assert result.addresses().item(0) % 64 == 0

        result[:] = 0

        for i in range(16):
            unif[i] = result.addresses()[i, i]

        drv.execute(code, unif.addresses()[0])

        tmud = np.arange(1, 16 * vec + 1, dtype=np.uint32).reshape(vec, 16)
        expected = np.zeros((16, 16), dtype=np.uint32)
        for i in range(4):
            expected.reshape(4, 4, 4, 4)[i, :, i, :] = np.vectorize(np.roll, signature="(n),()->(n)")(
                np.pad(tmud, [(0, 4 - vec), (0, 0)]).reshape(4, 4, 4)[:, i, :].T, np.arange(4)
            )

        # print()
        # print(result)
        # print(expected)

        assert (result == expected).all()


def test_tmu_vec3_naive_write() -> None:
    # result[:] <- 0 (result.addresses()[0, 0] must be aligned to 64 bytes)
    #
    # tmuc <- vec3 write
    #
    # tmud[0] <- [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
    # tmud[1] <- [17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]
    # tmud[2] <- [33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48]
    #
    # tmua <- result.addresses()[i, i]
    #
    # result =
    # [[ 1 17 33  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  2 18 34  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [35  0  3 19  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [20 36  0  4  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  5 21 37  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  6 22 38  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0 39  0  7 23  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0 24 40  0  8  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  9 25 41  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0 10 26 42  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0 43  0 11 27  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0 28 44  0 12  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 13 29 45  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0 14 30 46]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 47  0 15 31]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 32 48  0 16]]
    #
    vec = 3
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_naive_write, vec)
        result: Array[np.uint32] = drv.alloc((16, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)

        assert result.addresses().item(0) % 64 == 0

        result[:] = 0

        for i in range(16):
            unif[i] = result.addresses()[i, i]

        drv.execute(code, unif.addresses()[0])

        tmud = np.arange(1, 16 * vec + 1, dtype=np.uint32).reshape(vec, 16)
        expected = np.zeros((16, 16), dtype=np.uint32)
        for i in range(4):
            expected.reshape(4, 4, 4, 4)[i, :, i, :] = np.vectorize(np.roll, signature="(n),()->(n)")(
                np.pad(tmud, [(0, 4 - vec), (0, 0)]).reshape(4, 4, 4)[:, i, :].T, np.arange(4)
            )

        # print()
        # print(result)
        # print(expected)

        assert (result == expected).all()


def test_tmu_vec2_naive_write() -> None:
    # result[:] <- 0 (result.addresses()[0, 0] must be aligned to 64 bytes)
    #
    # tmuc <- vec2 write
    #
    # tmud[0] <- [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
    # tmud[1] <- [17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]
    #
    # tmua <- result.addresses()[i, i]
    #
    # result =
    # [[ 1 17  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  2 18  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  3 19  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [20  0  0  4  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  5 21  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  6 22  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  7 23  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0 24  0  0  8  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  9 25  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0 10 26  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0 11 27  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0 28  0  0 12  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 13 29  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0 14 30  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 15 31]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 32  0  0 16]]
    #
    vec = 2
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_naive_write, vec)
        result: Array[np.uint32] = drv.alloc((16, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)

        assert result.addresses().item(0) % 64 == 0

        result[:] = 0

        for i in range(16):
            unif[i] = result.addresses()[i, i]

        drv.execute(code, unif.addresses()[0])

        tmud = np.arange(1, 16 * vec + 1, dtype=np.uint32).reshape(vec, 16)
        expected = np.zeros((16, 16), dtype=np.uint32)
        for i in range(4):
            expected.reshape(4, 4, 4, 4)[i, :, i, :] = np.vectorize(np.roll, signature="(n),()->(n)")(
                np.pad(tmud, [(0, 4 - vec), (0, 0)]).reshape(4, 4, 4)[:, i, :].T, np.arange(4)
            )

        # print()
        # print(result)
        # print(expected)

        assert (result == expected).all()


@qpu
def qpu_tmu_vec_n_naive_read(asm: Assembly, vec_n: int) -> None:
    assert 2 <= vec_n <= 4

    eidx(rf1, cond="pushz")
    nop(sig=ldunifrf(rf2))
    mov(rf10, rf2, cond="ifa")
    for i in range(15):
        nop(sig=ldunifrf(rf2))
        sub(rf1, rf1, 1, cond="pushz")
        mov(rf10, rf2, cond="ifa")

    bnot(tmuc, 7 - vec_n)  # vec n read
    mov(tmua, rf10, sig=thrsw)
    nop()
    nop()
    for i in range(vec_n):
        nop(sig=ldtmu(rf[20 + i]))

    eidx(rf1, sig=ldunifrf(rf10))
    shl(rf1, rf1, 2)
    add(rf10, rf10, rf1)
    shl(rf2, 4, 4)
    for i in range(vec_n):
        mov(tmud, rf[20 + i])
        mov(tmua, rf10).add(rf10, rf10, rf2)

    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_vec4_naive_read() -> None:
    # result[:] <- 0 (result.addresses()[0, 0] must be aligned to 64 bytes)
    #
    # data =
    # [[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16]
    #  [ 17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32]
    #  [ 33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48]
    #  [ 49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64]
    #  [ 65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80]
    #  [ 81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96]
    #  [ 97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112]
    #  [113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128]
    #  [129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144]
    #  [145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160]
    #  [161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176]
    #  [177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192]
    #  [193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208]
    #  [209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224]
    #  [225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240]
    #  [241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256]]
    #
    # tmuc <- vec4 read
    # tmua <- data.addresses()[i, i]
    #
    # result[i] <- tmu[i]
    #
    # result =
    # [[  1  18  35  52  69  86 103 120 137 154 171 188 205 222 239 256]
    #  [  2  19  36  49  70  87 104 117 138 155 172 185 206 223 240 253]
    #  [  3  20  33  50  71  88 101 118 139 156 169 186 207 224 237 254]
    #  [  4  17  34  51  72  85 102 119 140 153 170 187 208 221 238 255]]
    #
    vec = 4
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_naive_read, vec)
        data: Array[np.uint32] = drv.alloc((16, 16), dtype=np.uint32)
        result: Array[np.uint32] = drv.alloc((vec, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16 + 1, dtype=np.uint32)

        assert result.addresses().item(0) % 64 == 0

        data[:] = np.arange(1, 16 * 16 + 1, dtype=np.uint32).reshape(16, 16)
        result[:] = 0

        for i in range(16):
            unif[i] = data.addresses()[i, i]
        unif[16] = result.addresses().item(0)

        drv.execute(code, unif.addresses()[0])

        expected = np.zeros((vec, 16), dtype=np.uint32)
        for i in range(4):
            expected.reshape(vec, 4, 4)[:, i, :] = np.vectorize(np.roll, signature="(n),()->(n)")(
                data.reshape(4, 4, 4, 4)[i, :, i, :], -np.arange(4)
            )[:, :vec].T

        # print()
        # print(data)
        # print(result)
        # print(expected)

        assert (result == expected).all()


def test_tmu_vec3_naive_read() -> None:
    # result[:] <- 0 (result.addresses()[0, 0] must be aligned to 64 bytes)
    #
    # data =
    # [[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16]
    #  [ 17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32]
    #  [ 33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48]
    #  [ 49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64]
    #  [ 65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80]
    #  [ 81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96]
    #  [ 97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112]
    #  [113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128]
    #  [129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144]
    #  [145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160]
    #  [161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176]
    #  [177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192]
    #  [193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208]
    #  [209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224]
    #  [225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240]
    #  [241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256]]
    #
    # tmuc <- vec3 read
    # tmua <- data.addresses()[i, i]
    #
    # result[i] <- tmu[i]
    #
    # result =
    # [[  1  18  35  52  69  86 103 120 137 154 171 188 205 222 239 256]
    #  [  2  19  36  49  70  87 104 117 138 155 172 185 206 223 240 253]
    #  [  3  20  33  50  71  88 101 118 139 156 169 186 207 224 237 254]]
    #
    vec = 3
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_naive_read, vec)
        data: Array[np.uint32] = drv.alloc((16, 16), dtype=np.uint32)
        result: Array[np.uint32] = drv.alloc((vec, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16 + 1, dtype=np.uint32)

        assert result.addresses().item(0) % 64 == 0

        data[:] = np.arange(1, 16 * 16 + 1, dtype=np.uint32).reshape(16, 16)
        result[:] = 0

        for i in range(16):
            unif[i] = data.addresses()[i, i]
        unif[16] = result.addresses().item(0)

        drv.execute(code, unif.addresses()[0])

        expected = np.zeros((vec, 16), dtype=np.uint32)
        for i in range(4):
            expected.reshape(vec, 4, 4)[:, i, :] = np.vectorize(np.roll, signature="(n),()->(n)")(
                data.reshape(4, 4, 4, 4)[i, :, i, :], -np.arange(4)
            )[:, :vec].T

        # print()
        # print(data)
        # print(result)
        # print(expected)

        assert (result == expected).all()


def test_tmu_vec2_naive_read() -> None:
    # result[:] <- 0 (result.addresses()[0, 0] must be aligned to 64 bytes)
    #
    # data =
    # [[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16]
    #  [ 17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32]
    #  [ 33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48]
    #  [ 49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64]
    #  [ 65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80]
    #  [ 81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96]
    #  [ 97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112]
    #  [113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128]
    #  [129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144]
    #  [145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160]
    #  [161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176]
    #  [177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192]
    #  [193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208]
    #  [209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224]
    #  [225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240]
    #  [241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256]]
    #
    # tmuc <- vec3 read
    # tmua <- data.addresses()[i, i]
    #
    # result[i] <- tmu[i]
    #
    # result =
    # [[  1  18  35  52  69  86 103 120 137 154 171 188 205 222 239 256]
    #  [  2  19  36  49  70  87 104 117 138 155 172 185 206 223 240 253]]
    #
    vec = 2
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_naive_read, vec)
        data: Array[np.uint32] = drv.alloc((16, 16), dtype=np.uint32)
        result: Array[np.uint32] = drv.alloc((vec, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16 + 1, dtype=np.uint32)

        assert result.addresses().item(0) % 64 == 0

        data[:] = np.arange(1, 16 * 16 + 1, dtype=np.uint32).reshape(16, 16)
        result[:] = 0

        for i in range(16):
            unif[i] = data.addresses()[i, i]
        unif[16] = result.addresses().item(0)

        drv.execute(code, unif.addresses()[0])

        expected = np.zeros((vec, 16), dtype=np.uint32)
        for i in range(4):
            expected.reshape(vec, 4, 4)[:, i, :] = np.vectorize(np.roll, signature="(n),()->(n)")(
                data.reshape(4, 4, 4, 4)[i, :, i, :], -np.arange(4)
            )[:, :vec].T

        # print()
        # print(data)
        # print(result)
        # print(expected)

        assert (result == expected).all()


@qpu
def qpu_tmu_vec_n_contiguous_write(asm: Assembly, vec_n: int) -> None:
    assert 2 <= vec_n <= 4

    eidx(rf1, cond="pushz")
    nop(sig=ldunifrf(rf2))
    mov(rf10, rf2, cond="ifa")
    for i in range(15):
        nop(sig=ldunifrf(rf2))
        sub(rf1, rf1, 1, cond="pushz")
        mov(rf10, rf2, cond="ifa")

    mov(rf1, 8)
    add(rf1, rf1, 8)
    eidx(rf20)
    add(rf20, rf20, 1)
    add(rf21, rf20, rf1)
    add(rf22, rf21, rf1)
    add(rf23, rf22, rf1)

    shr(rf11, rf10, 2)
    band(rf11, rf11, 3)
    shr(rf12, rf10, 4)
    add(rf12, rf12, 1)
    shl(rf12, rf12, 4)

    sub(null, rf11, 5 - vec_n, cond="pushn")
    b(R.skip_0, cond="allna")
    nop()
    nop()
    nop()
    bnot(tmuc, 7 - vec_n)
    for i in range(vec_n):
        mov(tmud, rf[20 + i])
    mov(tmua, rf10, cond="ifa")
    L.skip_0

    for n in range(1, vec_n):
        with namespace(f"ns{n}"):
            sub(null, rf11, (4 + n) - vec_n, cond="pushz")
            b(R.skip, cond="allna")
            nop()
            nop()
            nop()

            if vec_n - n > 1:
                bnot(tmuc, 7 - (vec_n - n))
            for i in range(vec_n - n):
                mov(tmud, rf[20 + i])
            mov(tmua, rf10, cond="ifa")

            if n > 1:
                bnot(tmuc, 7 - n)
            for i in range(n):
                mov(tmud, rf[20 + vec_n - n + i])
            mov(tmua, rf12, cond="ifa")

            L.skip

    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_vec4_contiguous_write() -> None:
    # result[:] <- 0
    #
    # tmud[0] <- [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
    # tmud[1] <- [17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]
    # tmud[2] <- [33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48]
    # tmud[3] <- [49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64]
    #
    # result =
    # [[ 1 17 33 49  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  2 18 34 50  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  3 19 35 51  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  4 20 36 52  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  5 21 37 53  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  6 22 38 54  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  7 23 39 55  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  8 24 40 56  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  9 25 41 57  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0 10 26 42 58  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0 11 27 43 59  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0 12 28 44 60  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 13 29 45 61  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0 14 30 46 62  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 15 31 47 63  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 16 32 48 64]]
    #
    vec = 4
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_contiguous_write, vec)
        result: Array[np.uint32] = drv.alloc((16, 19), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)

        result[:] = 0

        tmud = np.arange(1, 16 * vec + 1, dtype=np.uint32).reshape(vec, 16)
        expected = np.zeros((16, 19), dtype=np.uint32)
        for i in range(16):
            unif[i] = result.addresses()[i, i]
            expected[i, i : i + vec] = tmud[:, i]

        drv.execute(code, unif.addresses()[0])

        # print()
        # print(result)
        # print(expected)

        assert (result == expected).all()


def test_tmu_vec3_contiguous_write() -> None:
    # result[:] <- 0
    #
    # tmud[0] <- [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
    # tmud[1] <- [17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]
    # tmud[2] <- [33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48]
    #
    # result =
    # [[ 1 17 33  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  2 18 34  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  3 19 35  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  4 20 36  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  5 21 37  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  6 22 38  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  7 23 39  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  8 24 40  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  9 25 41  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0 10 26 42  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0 11 27 43  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0 12 28 44  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 13 29 45  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0 14 30 46  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 15 31 47  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 16 32 48  0]]
    #
    vec = 3
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_contiguous_write, vec)
        result: Array[np.uint32] = drv.alloc((16, 19), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)

        result[:] = 0

        tmud = np.arange(1, 16 * vec + 1, dtype=np.uint32).reshape(vec, 16)
        expected = np.zeros((16, 19), dtype=np.uint32)
        for i in range(16):
            unif[i] = result.addresses()[i, i]
            expected[i, i : i + vec] = tmud[:, i]

        drv.execute(code, unif.addresses()[0])

        # print()
        # print(result)
        # print(expected)

        assert (result == expected).all()


def test_tmu_vec2_contiguous_write() -> None:
    # result[:] <- 0
    #
    # tmud[0] <- [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
    # tmud[1] <- [17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]
    #
    # result =
    # [[ 1 17  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  2 18  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  3 19  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  4 20  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  5 21  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  6 22  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  7 23  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  8 24  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  9 25  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0 10 26  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0 11 27  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0 12 28  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 13 29  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0 14 30  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 15 31  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 16 32  0  0]]
    #
    vec = 2
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_contiguous_write, vec)
        result: Array[np.uint32] = drv.alloc((16, 19), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)

        result[:] = 0

        tmud = np.arange(1, 16 * vec + 1, dtype=np.uint32).reshape(vec, 16)
        expected = np.zeros((16, 19), dtype=np.uint32)
        for i in range(16):
            unif[i] = result.addresses()[i, i]
            expected[i, i : i + vec] = tmud[:, i]

        drv.execute(code, unif.addresses()[0])

        # print()
        # print(result)
        # print(expected)

        assert (result == expected).all()


@qpu
def qpu_tmu_vec_n_contiguous_read(asm: Assembly, vec_n: int) -> None:
    eidx(rf1, cond="pushz")
    nop(sig=ldunifrf(rf2))
    mov(rf10, rf2, cond="ifa")
    for i in range(15):
        nop(sig=ldunifrf(rf2))
        sub(rf1, rf1, 1, cond="pushz")
        mov(rf10, rf2, cond="ifa")

    shr(rf11, rf10, 2)
    band(rf11, rf11, 3)
    shr(rf12, rf10, 4)
    add(rf12, rf12, 1)
    shl(rf12, rf12, 4)

    for i in range(vec_n):
        mov(rf[20 + i], 0)

    sub(null, rf11, 5 - vec_n, cond="pushn")
    b(R.skip_0, cond="allna")
    nop()
    nop()
    nop()
    bnot(tmuc, 7 - vec_n)
    mov(tmua, rf10, cond="ifa", sig=thrsw)
    nop()
    nop()
    for i in range(vec_n):
        nop(sig=ldtmu(rf[30 + i]))
        mov(rf[20 + i], rf[30 + i], cond="ifa")
    L.skip_0

    for n in range(1, vec_n):
        with namespace(f"ns{n}"):
            sub(null, rf11, (4 + n) - vec_n, cond="pushz")
            b(R.skip, cond="allna")
            nop()
            nop()
            nop()

            if vec_n - n > 1:
                bnot(tmuc, 7 - (vec_n - n))
            mov(tmua, rf10, cond="ifa", sig=thrsw)
            nop()
            nop()
            for i in range(vec_n - n):
                nop(sig=ldtmu(rf[30 + i]))

            if n > 1:
                bnot(tmuc, 7 - n)
            mov(tmua, rf12, cond="ifa", sig=thrsw)
            nop()
            nop()
            for i in range(n):
                nop(sig=ldtmu(rf[30 + vec_n - n + i]))

            for i in range(vec_n):
                mov(rf[20 + i], rf[30 + i], cond="ifa")
            L.skip

    eidx(rf1, sig=ldunifrf(rf10))
    shl(rf1, rf1, 2)
    add(rf10, rf10, rf1)
    shl(rf2, 4, 4)
    for i in range(vec_n):
        mov(tmud, rf[20 + i])
        mov(tmua, rf10).add(rf10, rf10, rf2)

    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_vec4_contiguous_read() -> None:
    # result[:] <- 0
    #
    # data =
    # [[ 1  2  3  4  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  5  6  7  8  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  9 10 11 12  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0 13 14 15 16  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0 17 18 19 20  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0 21 22 23 24  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0 25 26 27 28  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0 29 30 31 32  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0 33 34 35 36  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0 37 38 39 40  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0 41 42 43 44  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0 45 46 47 48  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 49 50 51 52  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0 53 54 55 56  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 57 58 59 60  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 61 62 63 64  0]]
    #
    # result[i] <- tmu[i]
    #
    # result =
    # [[ 1  5  9 13 17 21 25 29 33 37 41 45 49 53 57 61]
    #  [ 2  6 10 14 18 22 26 30 34 38 42 46 50 54 58 62]
    #  [ 3  7 11 15 19 23 27 31 35 39 43 47 51 55 59 63]
    #  [ 4  8 12 16 20 24 28 32 36 40 44 48 52 56 60 64]]
    #
    vec = 4
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_contiguous_read, vec)
        data: Array[np.uint32] = drv.alloc((16, 20), dtype=np.uint32)
        result: Array[np.uint32] = drv.alloc((vec, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16 + 1, dtype=np.uint32)

        assert result.addresses().item(0) % 64 == 0

        expected = np.arange(1, 16 * vec + 1, dtype=np.uint32).reshape(16, vec).T
        data[:] = np.vectorize(np.roll, signature="(n),()->(n)")(
            np.pad(expected.T, [(0, 0), (0, 20 - vec)]), np.arange(16)
        )
        result[:] = 0

        for i in range(16):
            unif[i] = data.addresses()[i, i]
        unif[16] = result.addresses().item(0)

        drv.execute(code, unif.addresses()[0])

        # print()
        # print(data)
        # print(result)
        # print(expected)

        assert (result == expected).all()


def test_tmu_vec3_contiguous_read() -> None:
    # result[:] <- 0
    #
    # data =
    # [[ 1  2  3  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  4  5  6  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  7  8  9  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0 10 11 12  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0 13 14 15  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0 16 17 18  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0 19 20 21  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0 22 23 24  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0 25 26 27  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0 28 29 30  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0 31 32 33  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0 34 35 36  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 37 38 39  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0 40 41 42  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 43 44 45  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 46 47 48  0  0]]
    #
    # result[i] <- tmu[i]
    #
    # result =
    # [[ 1  4  7 10 13 16 19 22 25 28 31 34 37 40 43 46]
    #  [ 2  5  8 11 14 17 20 23 26 29 32 35 38 41 44 47]
    #  [ 3  6  9 12 15 18 21 24 27 30 33 36 39 42 45 48]]
    #
    vec = 3
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_contiguous_read, vec)
        data: Array[np.uint32] = drv.alloc((16, 20), dtype=np.uint32)
        result: Array[np.uint32] = drv.alloc((vec, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16 + 1, dtype=np.uint32)

        assert result.addresses().item(0) % 64 == 0

        expected = np.arange(1, 16 * vec + 1, dtype=np.uint32).reshape(16, vec).T
        data[:] = np.vectorize(np.roll, signature="(n),()->(n)")(
            np.pad(expected.T, [(0, 0), (0, 20 - vec)]), np.arange(16)
        )
        result[:] = 0

        for i in range(16):
            unif[i] = data.addresses()[i, i]
        unif[16] = result.addresses().item(0)

        drv.execute(code, unif.addresses()[0])

        # print()
        # print(data)
        # print(result)
        # print(expected)

        assert (result == expected).all()


def test_tmu_vec2_contiguous_read() -> None:
    # result[:] <- 0
    #
    # data =
    # [[ 1  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  3  4  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  5  6  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  7  8  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  9 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0 11 12  0  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0 13 14  0  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0 15 16  0  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0 17 18  0  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0 19 20  0  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0 21 22  0  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0 23 24  0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0 25 26  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0 27 28  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 29 30  0  0  0  0]
    #  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 31 32  0  0  0]]
    #
    # result[i] <- tmu[i]
    #
    # result =
    # [[ 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31]
    #  [ 2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32]]
    #
    vec = 2
    with Driver() as drv:
        code = drv.program(qpu_tmu_vec_n_contiguous_read, vec)
        data: Array[np.uint32] = drv.alloc((16, 20), dtype=np.uint32)
        result: Array[np.uint32] = drv.alloc((vec, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(16 + 1, dtype=np.uint32)

        assert result.addresses().item(0) % 64 == 0

        expected = np.arange(1, 16 * vec + 1, dtype=np.uint32).reshape(16, vec).T
        data[:] = np.vectorize(np.roll, signature="(n),()->(n)")(
            np.pad(expected.T, [(0, 0), (0, 20 - vec)]), np.arange(16)
        )
        result[:] = 0

        for i in range(16):
            unif[i] = data.addresses()[i, i]
        unif[16] = result.addresses().item(0)

        drv.execute(code, unif.addresses()[0])

        # print()
        # print(data)
        # print(result)
        # print(expected)

        assert (result == expected).all()
