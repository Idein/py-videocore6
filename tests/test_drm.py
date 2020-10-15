
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


from videocore6.drm_v3d import DRM_V3D


def test_get_param():
    print()

    with DRM_V3D() as drm:

        uifcfg       = drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_UIFCFG)
        hub_ident1   = drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_HUB_IDENT1)
        hub_ident2   = drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_HUB_IDENT2)
        hub_ident3   = drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_HUB_IDENT3)
        core0_ident0 = drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_CORE0_IDENT0)
        core0_ident1 = drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_CORE0_IDENT1)
        core0_ident2 = drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_CORE0_IDENT2)
        supports_tfu = drm.v3d_get_param(DRM_V3D.V3D_PARAM_SUPPORTS_TFU)
        supports_csd = drm.v3d_get_param(DRM_V3D.V3D_PARAM_SUPPORTS_CSD)

        print(f'uifcfg:       {uifcfg:#010x}')
        print(f'hub_ident1:   {hub_ident1:#010x}')
        print(f'hub_ident2:   {hub_ident2:#010x}')
        print(f'hub_ident3:   {hub_ident3:#010x}')
        print(f'core0_ident0: {core0_ident0:#010x}')
        print(f'core0_ident1: {core0_ident1:#010x}')
        print(f'core0_ident2: {core0_ident2:#010x}')
        print(f'supports_tfu: {supports_tfu:#010x}')
        print(f'supports_csd: {supports_csd:#010x}')

    print('Consult /sys/kernel/debug/dri/0/v3d_regs for more information')


def test_alloc():
    print()

    size = pow(2, 24)

    with DRM_V3D() as drm:

        handle, phyaddr = drm.v3d_create_bo(size)
        offset = drm.v3d_mmap_bo(handle)

        print(f'size    = {size:#010x}')
        print(f'handle  = {handle:#010x}')
        print(f'phyaddr = {phyaddr:#010x}')
        print(f'offset  = {offset:#010x}')
