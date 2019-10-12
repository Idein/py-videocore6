
__version__ = '0.0.0'

import struct


def float_to_int(f):
    return struct.unpack('i', struct.pack('f', f))[0]


def int_to_float(i):
    return struct.unpack('f', struct.pack('i', i))[0]


def int_to_uint(i):
    return struct.unpack('I', struct.pack('i', i))[0]
