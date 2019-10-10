
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
