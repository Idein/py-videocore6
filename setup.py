
import platform

from setuptools import setup, Extension

from videocore6 import __version__ as version


ext_modules = []

if platform.machine() == 'armv7l':
    ext_modules.append(Extension('videocore6.readwrite4',
                                 sources = ['videocore6/readwrite4.c']))

setup(
        name = 'py-videocore6',
        packages = [
                'videocore6',
        ],
        version = version,
        description = 'Python library for GPGPU programming on Raspberry Pi 4',
        author = 'Sugizaki Yukimasa',
        author_email = 'ysugi@idein.jp',
        install_requires = [
                'ioctl-opt >= 1.2',
                'numpy',
        ],
        ext_modules = ext_modules,
        python_requires = '~= 3.7',  # for f-string.
)
