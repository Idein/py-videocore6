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
import sys
import warnings
from collections.abc import Callable
from types import TracebackType
from typing import IO, Any, Concatenate, ParamSpec, Self, TypeVar

import numpy as np
import numpy.typing as npt

from _videocore6.assembler import Assembly, assemble
from _videocore6.drm_v3d import DRM_V3D

DEFAULT_CODE_AREA_SIZE = 1024 * 1024
DEFAULT_DATA_AREA_SIZE = 32 * 1024 * 1024


def to_be_removed_local_invocation_defaulting(local_invocation: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if local_invocation is None:
        warnings.warn(
            (
                "local_invocation is not specified; "
                "using (16, 1, 1) as a fallback, "
                "but correct behavior is not guaranteed."
            ),
            UserWarning,
        )
        return (16, 1, 1)
    return local_invocation


class DriverError(Exception):
    pass


T = TypeVar("T", bound=np.generic)  # Type variable for numpy array types
P = ParamSpec("P")


class Array(npt.NDArray[T]):
    address: int

    def __new__(cls: type["Array[T]"], *args: Any, phyaddr: int, **kwargs: Any) -> "Array[T]":
        obj = super().__new__(cls, *args, **kwargs)
        obj.address = phyaddr
        return obj

    def addresses(self: Self) -> npt.NDArray[np.uint32]:
        return np.arange(
            self.address,
            self.address + self.nbytes,
            self.itemsize,
            np.uint32,
        ).reshape(self.shape)


class Memory:
    drm: DRM_V3D
    size: int
    handle: int | None
    phyaddr: int | None
    buffer: mmap.mmap | None

    def __init__(self: Self, drm: DRM_V3D, size: int) -> None:
        self.drm = drm
        self.size = size
        self.handle = None  # Handle of BO for V3D DRM
        self.phyaddr = None  # Physical address used in QPU
        self.buffer = None  # mmap object of the memory area

        try:
            self.handle, self.phyaddr = drm.v3d_create_bo(size)
            offset = drm.v3d_mmap_bo(self.handle)
            fd = drm.fd
            if fd is None:
                raise DriverError("Failed to get file descriptor")
            self.buffer = mmap.mmap(
                fileno=fd,
                length=size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset,
            )

        except Exception as e:
            self.close()
            raise e

    def close(self: Self) -> None:
        if self.buffer is not None:
            self.buffer.close()

        if self.handle is not None:
            self.drm.gem_close(self.handle)

        self.handle = None
        self.phyaddr = None
        self.buffer = None


class Dispatcher:
    drm: DRM_V3D
    bo_handles: npt.NDArray[np.uint32]
    timeout_sec: int

    def __init__(self: Self, drm: DRM_V3D, bo_handles: npt.NDArray[np.uint32], timeout_sec: int = 10) -> None:
        self.drm = drm
        self.bo_handles = bo_handles
        self.timeout_sec = timeout_sec

    def __enter__(self: Self) -> Self:
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: type[TracebackType] | None,
    ) -> None:
        for bo_handle in self.bo_handles:
            self.drm.v3d_wait_bo(bo_handle, timeout_ns=int(self.timeout_sec / 1e-9))

    def dispatch(
        self: Self,
        code: Array[np.uint64],
        local_invocation: tuple[int, int, int] | None = None,
        uniforms: int | None = None,
        workgroup: tuple[int, int, int] = (1, 1, 1),
        wgs_per_sg: int = 16,
        thread: int = 1,
        propagate_nan: bool = False,
        single_seg: bool = False,
        threading: bool = False,
    ) -> None:
        wg_x, wg_y, wg_z = workgroup
        li_x, li_y, li_z = to_be_removed_local_invocation_defaulting(local_invocation)
        wg_size = li_x * li_y * li_z

        def roundup(n: int, d: int) -> int:
            return (n + d - 1) // d

        self.drm.v3d_submit_csd(
            cfg=(
                # WGS X, Y, Z and settings
                wg_x << 16,
                wg_y << 16,
                wg_z << 16,
                ((roundup(wgs_per_sg * wg_size, 16) - 1) << 12) | (wgs_per_sg << 8) | (wg_size & 0xFF),
                # Number of batches minus 1
                thread - 1,
                # Shader address, pnan, singleseg, threading
                code.addresses()[0] | (int(propagate_nan) << 2) | (int(single_seg) << 1) | int(threading),
                # Uniforms address
                uniforms if uniforms is not None else 0,
            ),
            # Not used in the driver.
            coef=(0, 0, 0, 0),
            bo_handles=self.bo_handles.ctypes.data,
            bo_handle_count=len(self.bo_handles),
            in_sync=0,
            out_sync=0,
        )


class Driver:
    _code_area_size: int
    _data_area_size: int
    _code_area_base: int
    _data_area_base: int
    _code_pos: int
    _data_pos: int
    _drm: DRM_V3D | None
    _memory: Memory | None
    _bo_handles: npt.NDArray[np.uint32] | None

    def __init__(
        self: Self,
        *,
        code_area_size: int = DEFAULT_CODE_AREA_SIZE,
        data_area_size: int = DEFAULT_DATA_AREA_SIZE,
    ) -> None:
        self.code_area_size = code_area_size
        self.data_area_size = data_area_size
        total_size = self.code_area_size + self.data_area_size
        self.code_area_base = 0
        self.data_area_base = self.code_area_base + self.code_area_size
        self.code_pos = self.code_area_base
        self.data_pos = self.data_area_base

        self.drm = None
        self.memory = None
        self.bo_handles = None

        try:
            self.drm = DRM_V3D()

            self.memory = Memory(self.drm, total_size)

            self.bo_handles = np.array([self.memory.handle], dtype=np.uint32)

        except Exception as e:
            self.close()
            raise e

    def close(self: Self) -> None:
        if self.memory is not None:
            self.memory.close()

        if self.drm is not None:
            self.drm.close()

        self.drm = None
        self.memory = None
        self.bo_handles = None

    def __enter__(self: Self) -> Self:
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: type[TracebackType] | None,
    ) -> bool:
        self.close()
        return exc_type is None

    def alloc(self: Self, *args: Any, **kwargs: Any) -> Array[T]:
        if self.memory is None:
            raise DriverError("Driver is closed")
        if self.memory.phyaddr is None:
            raise DriverError("Memory is not initialized")

        offset = self.data_pos
        kwargs["phyaddr"] = self.memory.phyaddr + offset
        kwargs["buffer"] = self.memory.buffer
        kwargs["offset"] = offset

        arr = Array[T](*args, **kwargs)

        self.data_pos += arr.nbytes
        if self.data_pos > self.data_area_base + self.data_area_size:
            raise DriverError("Data too large")

        return arr

    def dump_code(self: Self, code: list[int], *, file: IO[str] = sys.stdout) -> None:
        for insn in code:
            print(f"{insn:#018x}", file=file)

    def dump_program(
        self: Self,
        prog: Callable[Concatenate[Assembly, P], Any],
        file: IO[str] = sys.stdout,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        self.dump_code(assemble(prog, *args, **kwargs), file=file)

    def program(
        self: Self,
        prog: Callable[Concatenate[Assembly, P], Any] | list[int],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Array[np.uint64]:
        if hasattr(prog, "__call__"):
            asm = assemble(prog, *args, **kwargs)
        elif isinstance(prog, list):
            asm = prog
        else:
            raise RuntimeError("unreachable")

        if self.memory is None:
            raise DriverError("Driver is closed")
        if self.memory.phyaddr is None:
            raise DriverError("Memory is not initialized")

        offset = self.code_pos
        code = Array[np.uint64](
            shape=len(asm),
            dtype=np.uint64,
            phyaddr=self.memory.phyaddr + offset,
            buffer=self.memory.buffer,
            offset=offset,
        )

        self.code_pos += code.nbytes
        if self.code_pos > self.code_area_base + self.code_area_size:
            raise DriverError("Code too large")

        code[:] = asm

        return code

    def compute_shader_dispatcher(self: Self, timeout_sec: int = 10) -> Dispatcher:
        if self.drm is None:
            raise DriverError("Driver is closed")
        if self.bo_handles is None:
            raise DriverError("BO handles are not initialized")
        return Dispatcher(self.drm, self.bo_handles, timeout_sec=timeout_sec)

    def execute(
        self: Self,
        code: Array[np.uint64],
        local_invocation: tuple[int, int, int] | None = None,
        uniforms: int | None = None,
        timeout_sec: int = 10,
        workgroup: tuple[int, int, int] = (1, 1, 1),
        wgs_per_sg: int = 16,
        thread: int = 1,
        propagate_nan: bool = False,
        single_seg: bool = False,
        threading: bool = False,
    ) -> None:
        with self.compute_shader_dispatcher(timeout_sec) as csd:
            csd.dispatch(
                code,
                local_invocation=to_be_removed_local_invocation_defaulting(local_invocation),
                uniforms=uniforms,
                workgroup=workgroup,
                wgs_per_sg=wgs_per_sg,
                thread=thread,
                propagate_nan=propagate_nan,
                single_seg=single_seg,
                threading=threading,
            )
