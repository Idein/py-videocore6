
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

import subprocess
from ctypes import cdll
import numpy as np

class BenchHelper(object):

    def __init__(self, path = './libbench_helper.so'):

        try:
            self.lib = cdll.LoadLibrary(path)
        except OSError:
            subprocess.run(f'gcc -O2 -shared -fPIC -o {path} -xc -'.split(), text=True,
                           input='''
#include <stdint.h>
void wait_address(uint32_t volatile * p) {
    while(p[0] == 0){}
}
'''
            )
            self.lib = cdll.LoadLibrary(path)


        self.lib.wait_address.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.uint32, shape=(1,), flags="C_CONTIGUOUS"),
        ]

    def wait_address(self, done):
        self.lib.wait_address(done)
