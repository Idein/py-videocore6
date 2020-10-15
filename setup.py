
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


import platform

from setuptools import setup, Extension

from videocore6 import __version__ as version


ext_modules = []

if platform.machine() in ['armv7l', 'aarch64']:
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
