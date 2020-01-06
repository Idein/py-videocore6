
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
