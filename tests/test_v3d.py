
from videocore6.drm_v3d import DRM_V3D
from videocore6.v3d import *


def test_v3d_regs():

    with DRM_V3D() as drm:

        try:

            with RegisterMapping() as regmap:

                assert regmap[HUB_UIFCFG] \
                        == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_UIFCFG)

                assert regmap[HUB_IDENT1] \
                        == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_HUB_IDENT1)

                assert regmap[HUB_IDENT2] \
                        == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_HUB_IDENT2)

                assert regmap[HUB_IDENT3] \
                        == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_HUB_IDENT3)

                assert regmap[CORE_IDENT0, 0] \
                        == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_CORE0_IDENT0)

                assert regmap[CORE_IDENT1, 0] \
                        == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_CORE0_IDENT1)

                assert regmap[CORE_IDENT2, 0] \
                        == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_CORE0_IDENT2)

        except PermissionError:

            print('Skipping tests because of a lack of root privilege')
