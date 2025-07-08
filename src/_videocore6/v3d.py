# Copyright (c) 2014-2018 Broadcom
# Copyright (c) 2019-2020 Idein Inc.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
# Street, Fifth Floor, Boston, MA 02110-1301 USA.
import mmap
import os
from ctypes import c_uint32, c_void_p, cdll
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import TracebackType
from typing import Self, cast

import numpy as np


class HubRegister:
    offset: int

    def __init__(self: Self, offset: int) -> None:
        self.offset = offset


class PerCoreRegister:
    offset: int

    def __init__(self: Self, offset: int) -> None:
        self.offset = offset


class HubField:
    register: HubRegister
    mask: int
    shift: int

    def __init__(self: Self, register: HubRegister, high: int, low: int) -> None:
        assert isinstance(register, HubRegister)
        self.register = register
        self.mask = ((1 << (high - low + 1)) - 1) << low
        self.shift = low


class PerCoreField:
    register: PerCoreRegister
    mask: int
    shift: int

    def __init__(self: Self, register: PerCoreRegister, high: int, low: int) -> None:
        assert isinstance(register, PerCoreRegister)
        self.register = register
        self.mask = ((1 << (high - low + 1)) - 1) << low
        self.shift = low


# V3D register definitions derived from linux/drivers/gpu/drm/v3d/v3d_regs.h

HUB_AXICFG = HubRegister(0x00000)

HUB_UIFCFG = HubRegister(0x00004)

HUB_IDENT0 = HubRegister(0x00008)

HUB_IDENT1 = HubRegister(0x0000C)
HUB_IDENT1_WITH_MSO = HubField(HUB_IDENT1, 19, 19)
HUB_IDENT1_WITH_TSY = HubField(HUB_IDENT1, 18, 18)
HUB_IDENT1_WITH_TFU = HubField(HUB_IDENT1, 17, 17)
HUB_IDENT1_WITH_L3C = HubField(HUB_IDENT1, 16, 16)
HUB_IDENT1_NHOSTS = HubField(HUB_IDENT1, 15, 12)
HUB_IDENT1_NCORES = HubField(HUB_IDENT1, 11, 8)
HUB_IDENT1_REV = HubField(HUB_IDENT1, 7, 4)
HUB_IDENT1_TVER = HubField(HUB_IDENT1, 3, 0)

HUB_IDENT2 = HubRegister(0x00010)
HUB_IDENT2_WITH_MMU = HubField(HUB_IDENT2, 8, 8)
HUB_IDENT2_L3C_NKB = HubField(HUB_IDENT2, 7, 0)

HUB_IDENT3 = HubRegister(0x00014)
HUB_IDENT3_IPREV = HubField(HUB_IDENT3, 15, 8)
HUB_IDENT3_IPIDX = HubField(HUB_IDENT3, 7, 0)

HUB_TFU_CS = HubRegister(0x00400)


CORE_IDENT0 = PerCoreRegister(0x00000)
CORE_IDENT0_VER = PerCoreField(CORE_IDENT0, 31, 24)

CORE_IDENT1 = PerCoreRegister(0x00004)
CORE_IDENT1_VPM_SIZE = PerCoreField(CORE_IDENT1, 31, 28)
CORE_IDENT1_NSEM = PerCoreField(
    CORE_IDENT1,
    23,
    16,
)
CORE_IDENT1_NTMU = PerCoreField(CORE_IDENT1, 15, 12)
CORE_IDENT1_QUPS = PerCoreField(CORE_IDENT1, 11, 8)
CORE_IDENT1_NSLC = PerCoreField(CORE_IDENT1, 7, 4)
CORE_IDENT1_REV = PerCoreField(CORE_IDENT1, 3, 0)

CORE_IDENT2 = PerCoreRegister(0x00008)
CORE_IDENT2_BCG = PerCoreField(CORE_IDENT2, 28, 28)

CORE_MISCCFG = PerCoreRegister(0x00018)
CORE_MISCCFG_QRMAXCNT = PerCoreField(CORE_MISCCFG, 3, 1)
CORE_MISCCFG_OVRTMUOUT = PerCoreField(CORE_MISCCFG, 0, 0)

CORE_L2CACTL = PerCoreRegister(0x00020)
CORE_L2CACTL_L2CCLR = PerCoreField(CORE_L2CACTL, 2, 2)
CORE_L2CACTL_L2CDIS = PerCoreField(CORE_L2CACTL, 1, 1)
CORE_L2CACTL_L2CENA = PerCoreField(CORE_L2CACTL, 0, 0)

CORE_SLCACTL = PerCoreRegister(0x00024)
CORE_SLCACTL_TVCCS = PerCoreField(CORE_SLCACTL, 27, 24)
CORE_SLCACTL_TDCCS = PerCoreField(CORE_SLCACTL, 19, 16)
CORE_SLCACTL_UCC = PerCoreField(CORE_SLCACTL, 11, 8)
CORE_SLCACTL_ICC = PerCoreField(CORE_SLCACTL, 3, 0)

CORE_PCTR_0_EN = PerCoreRegister(0x00650)
CORE_PCTR_0_CLR = PerCoreRegister(0x00654)
CORE_PCTR_0_OVERFLOW = PerCoreRegister(0x00658)

g = globals()

for i in range(0, 32, 4):
    name = f"CORE_PCTR_0_SRC_{i}_{i + 3}"
    g[name] = PerCoreRegister(0x00660 + i)
    g[name + f"_S{i + 3}"] = PerCoreField(g[name], 30, 24)
    g[name + f"_S{i + 2}"] = PerCoreField(g[name], 22, 16)
    g[name + f"_S{i + 1}"] = PerCoreField(g[name], 14, 8)
    g[name + f"_S{i + 0}"] = PerCoreField(g[name], 6, 0)
    g[f"CORE_PCTR_0_SRC_{i + 3}"] = PerCoreField(g[name], 30, 24)
    g[f"CORE_PCTR_0_SRC_{i + 2}"] = PerCoreField(g[name], 22, 16)
    g[f"CORE_PCTR_0_SRC_{i + 1}"] = PerCoreField(g[name], 14, 8)
    g[f"CORE_PCTR_0_SRC_{i + 0}"] = PerCoreField(g[name], 6, 0)

for i in range(32):
    g[f"CORE_PCTR_0_PCTR{i}"] = PerCoreRegister(0x00680 + 4 * i)

del g, i

CORE_PCTR_CYCLE_COUNT = 32


class RegisterMapping:
    def __init__(self: Self) -> None:
        stem = Path(__file__).parent / "readwrite4"
        for suffix in EXTENSION_SUFFIXES:
            try:
                lib = cdll.LoadLibrary(str(stem.with_suffix(suffix)))
            except OSError:
                continue
            else:
                break
        else:
            raise Exception("readwrite4 library is not found." + " Your installation seems to be broken.")

        self.read4 = lib.read4
        self.write4 = lib.write4
        del stem, lib

        self.read4.argtypes = [c_void_p]
        self.read4.restype = c_uint32
        self.write4.argtypes = [c_void_p, c_uint32]
        self.write4.restype = None

        fd = os.open("/dev/mem", os.O_RDWR)

        # XXX: Should use bcm_host_get_peripheral_address for the base address
        # on userland, and consult /proc/device-tree/__symbols__/v3d and then
        # /proc/device-tree/v3dbus/v3d@7ec04000/{reg-names,reg} for the offsets
        # in the future.

        self.map_hub = mmap.mmap(
            offset=0xFEC00000, length=0x4000, fileno=fd, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
        )
        self.ptr_hub = np.frombuffer(self.map_hub).ctypes.data

        self.ncores = 1
        self.map_cores: list[mmap.mmap] = []
        self.ptr_cores: list[int] = []
        for core in range(self.ncores):
            m = mmap.mmap(
                offset=0xFEC04000 + 0x4000 * core,
                length=0x4000,
                fileno=fd,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            self.map_cores.append(m)
            self.ptr_cores.append(np.frombuffer(m).ctypes.data)

        os.close(fd)

    def __enter__(self: Self) -> Self:
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: type[TracebackType] | None,
    ) -> None:
        pass

    def _get_ptr(self: Self, key: HubField | PerCoreField | HubRegister | PerCoreRegister, core: int | None) -> int:
        if isinstance(key, HubField | PerCoreField):
            return self._get_ptr(key.register, core)
        elif isinstance(key, HubRegister):
            assert core is None
            return self.ptr_hub + key.offset
        elif isinstance(key, PerCoreRegister):
            assert core is not None
            return self.ptr_cores[core] + key.offset

    def __getitem__(
        self: Self,
        key: tuple[HubField | PerCoreField | HubRegister | PerCoreRegister, int | None]
        | HubField
        | PerCoreField
        | HubRegister
        | PerCoreRegister,
    ) -> int:
        core = None
        if isinstance(key, tuple):
            key, core = key
        assert isinstance(key, HubField | PerCoreField | HubRegister | PerCoreRegister)

        v = cast(int, self.read4(self._get_ptr(key, core)))

        if isinstance(key, HubField | PerCoreField):
            v = (v & key.mask) >> key.shift

        return v

    def __setitem__(
        self: Self,
        key: tuple[HubField | PerCoreField | HubRegister | PerCoreRegister, int | None]
        | HubField
        | PerCoreField
        | HubRegister
        | PerCoreRegister,
        value: int,
    ) -> None:
        core = None
        if isinstance(key, tuple):
            key, core = key
        assert isinstance(key, HubField | PerCoreField | HubRegister | PerCoreRegister)

        if isinstance(key, HubField | PerCoreField):
            value = (self[key.register, core] & ~key.mask) | ((value << key.shift) & key.mask)

        self.write4(self._get_ptr(key, core), value)


class PerformanceCounter:
    _PCTR_SRCs = [globals()[f"CORE_PCTR_0_SRC_{_}"] for _ in range(32)]
    _PCTRs = [globals()[f"CORE_PCTR_0_PCTR{_}"] for _ in range(32)]

    def __init__(self: Self, regmap: RegisterMapping, srcs: list[int]) -> None:
        self.regmap = regmap
        self.srcs = srcs
        self.core = 0  # Sufficient for now.
        self.mask = (1 << len(self.srcs)) - 1

    def __enter__(self: Self) -> Self:
        self.regmap[CORE_PCTR_0_EN, self.core] = 0

        for i in range(len(self.srcs)):
            self.regmap[self._PCTR_SRCs[i], self.core] = self.srcs[i]

        self.regmap[CORE_PCTR_0_CLR, self.core] = self.mask
        self.regmap[CORE_PCTR_0_OVERFLOW, self.core] = self.mask
        self.regmap[CORE_PCTR_0_EN, self.core] = self.mask

        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: type[TracebackType] | None,
    ) -> None:
        self.regmap[CORE_PCTR_0_EN, self.core] = 0
        self.regmap[CORE_PCTR_0_CLR, self.core] = self.mask
        self.regmap[CORE_PCTR_0_OVERFLOW, self.core] = self.mask

    def result(self: Self) -> list[int]:
        return [self.regmap[self._PCTRs[i], self.core] for i in range(len(self.srcs))]
