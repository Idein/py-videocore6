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
"""V3D ioctl wrapper."""

import os
from ctypes import Structure, c_uint32, c_uint64
from fcntl import ioctl
from types import TracebackType
from typing import Final, Self

from ioctl_opt import IOW, IOWR


class DRM_V3D:  # noqa: N801
    fd: int | None

    def __init__(self: Self, path: str = "/dev/dri/by-path/platform-fec00000.v3d-card") -> None:
        self.fd = os.open(path, os.O_RDWR)

    def close(self: Self) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.fd = None

    def __enter__(self: Self) -> Self:
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: type[TracebackType] | None,
    ) -> bool:
        self.close()
        return exc_value is None

    # Derived from linux/include/uapi/drm/drm.h
    DRM_IOCTL_BASE: Final[int] = ord("d")
    DRM_COMMAND_BASE: Final[int] = 0x40
    DRM_GEM_CLOSE: Final[int] = 0x09

    # Derived from linux/include/uapi/drm/v3d_drm.h
    DRM_V3D_WAIT_BO: Final[int] = DRM_COMMAND_BASE + 0x01
    DRM_V3D_CREATE_BO: Final[int] = DRM_COMMAND_BASE + 0x02
    DRM_V3D_MMAP_BO: Final[int] = DRM_COMMAND_BASE + 0x03
    DRM_V3D_GET_PARAM: Final[int] = DRM_COMMAND_BASE + 0x04
    DRM_V3D_SUBMIT_CSD: Final[int] = DRM_COMMAND_BASE + 0x07

    V3D_PARAM_V3D_UIFCFG: Final[int] = 0
    V3D_PARAM_V3D_HUB_IDENT1: Final[int] = 1
    V3D_PARAM_V3D_HUB_IDENT2: Final[int] = 2
    V3D_PARAM_V3D_HUB_IDENT3: Final[int] = 3
    V3D_PARAM_V3D_CORE0_IDENT0: Final[int] = 4
    V3D_PARAM_V3D_CORE0_IDENT1: Final[int] = 5
    V3D_PARAM_V3D_CORE0_IDENT2: Final[int] = 6
    V3D_PARAM_SUPPORTS_TFU: Final[int] = 7
    V3D_PARAM_SUPPORTS_CSD: Final[int] = 8

    class st_gem_close(Structure):  # noqa: N801
        _fields_ = [
            ("handle", c_uint32),
            ("pad", c_uint32),
        ]

    class st_v3d_wait_bo(Structure):  # noqa: N801
        _fields_ = [
            ("handle", c_uint32),
            ("pad", c_uint32),
            ("timeout_ns", c_uint64),
        ]

    class st_v3d_create_bo(Structure):  # noqa: N801
        _fields_ = [
            ("size", c_uint32),
            ("flags", c_uint32),
            ("handle", c_uint32),
            ("offset", c_uint32),
        ]

    class st_v3d_mmap_bo(Structure):  # noqa: N801
        _fields_ = [
            ("handle", c_uint32),
            ("flags", c_uint32),
            ("offset", c_uint64),
        ]

    class st_v3d_get_param(Structure):  # noqa: N801
        _fields_ = [
            ("param", c_uint32),
            ("pad", c_uint32),
            ("value", c_uint64),
        ]

    class st_v3d_submit_csd(Structure):  # noqa: N801
        _fields_ = [
            ("cfg", c_uint32 * 7),
            ("coef", c_uint32 * 4),
            ("bo_handles", c_uint64),
            ("bo_handle_count", c_uint32),
            ("in_sync", c_uint32),
            ("out_sync", c_uint32),
        ]

    IOCTL_GEM_CLOSE: Final[int] = IOW(DRM_IOCTL_BASE, DRM_GEM_CLOSE, st_gem_close)

    IOCTL_V3D_WAIT_BO: Final[int] = IOWR(DRM_IOCTL_BASE, DRM_V3D_WAIT_BO, st_v3d_wait_bo)
    IOCTL_V3D_CREATE_BO: Final[int] = IOWR(DRM_IOCTL_BASE, DRM_V3D_CREATE_BO, st_v3d_create_bo)
    IOCTL_V3D_MMAP_BO: Final[int] = IOWR(DRM_IOCTL_BASE, DRM_V3D_MMAP_BO, st_v3d_mmap_bo)
    IOCTL_V3D_GET_PARAM: Final[int] = IOWR(DRM_IOCTL_BASE, DRM_V3D_GET_PARAM, st_v3d_get_param)
    IOCTL_V3D_SUBMIT_CSD: Final[int] = IOW(DRM_IOCTL_BASE, DRM_V3D_SUBMIT_CSD, st_v3d_submit_csd)

    def gem_close(self: Self, handle: int) -> None:
        if self.fd is None:
            raise ValueError("already closed")
        if handle < 0 or 0xFFFFFFFF < handle:
            raise ValueError("handle is out of range")
        st = self.st_gem_close(
            handle=handle,
            pad=0,
        )
        ioctl(self.fd, self.IOCTL_GEM_CLOSE, st)

    def v3d_wait_bo(self: Self, handle: int, timeout_ns: int) -> None:
        if self.fd is None:
            raise ValueError("already closed")
        if handle < 0 or 0xFFFFFFFF < handle:
            raise ValueError("handle is out of range")
        if timeout_ns < 0 or 0xFFFFFFFFFFFFFFFF < timeout_ns:
            raise ValueError("timeout_ns is out of range")
        st = self.st_v3d_wait_bo(
            handle=handle,
            pad=0,
            timeout_ns=timeout_ns,
        )
        ioctl(self.fd, self.IOCTL_V3D_WAIT_BO, st)

    def v3d_create_bo(self: Self, size: int, flags: int = 0) -> tuple[int, int]:
        if self.fd is None:
            raise ValueError("already closed")
        if size < 0 or 0xFFFFFFFF < size:
            raise ValueError("size is out of range")
        if flags < 0 or 0xFFFFFFFF < flags:
            raise ValueError("flags is out of range")
        st = self.st_v3d_create_bo(
            size=size,
            flags=flags,
            handle=0,
            offset=0,
        )
        ioctl(self.fd, self.IOCTL_V3D_CREATE_BO, st)
        return st.handle, st.offset

    def v3d_mmap_bo(self: Self, handle: int, flags: int = 0) -> int:
        if self.fd is None:
            raise ValueError("already closed")
        if handle < 0 or 0xFFFFFFFF < handle:
            raise ValueError("handle is out of range")
        if flags < 0 or 0xFFFFFFFF < flags:
            raise ValueError("flags is out of range")
        st = self.st_v3d_mmap_bo(
            handle=handle,
            flags=flags,
            offset=0,
        )
        ioctl(self.fd, self.IOCTL_V3D_MMAP_BO, st)
        return int(st.offset)

    def v3d_get_param(self: Self, param: int) -> int:
        if self.fd is None:
            raise ValueError("already closed")
        if param < 0 or 0xFFFFFFFF < param:
            raise ValueError("param is out of range")
        st = self.st_v3d_get_param(
            param=param,
            pad=0,
            value=0,
        )
        ioctl(self.fd, self.IOCTL_V3D_GET_PARAM, st)
        return int(st.value)

    def v3d_submit_csd(
        self: Self,
        cfg: tuple[int, int, int, int, int, int, int],
        coef: tuple[int, int, int, int],
        bo_handles: int,
        bo_handle_count: int,
        in_sync: int,
        out_sync: int,
    ) -> None:
        if self.fd is None:
            raise ValueError("already closed")
        st = self.st_v3d_submit_csd(
            # XXX: Dirty hack!
            cfg=(c_uint32 * 7)(*cfg),
            coef=(c_uint32 * 4)(*coef),
            bo_handles=bo_handles,
            bo_handle_count=bo_handle_count,
            in_sync=in_sync,
            out_sync=out_sync,
        )
        ioctl(self.fd, self.IOCTL_V3D_SUBMIT_CSD, st)
