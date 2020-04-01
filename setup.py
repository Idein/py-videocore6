
from setuptools import setup, Extension

from videocore6 import __version__ as version


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
        ext_modules = [
                Extension(
                        'videocore6.readwrite4',
                        sources = [
                                'videocore6/readwrite4.c',
                        ],
                ),
        ],
        python_requires = '~= 3.7',  # for f-string.
)
