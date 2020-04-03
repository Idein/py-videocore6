
__version__ = '0.0.0'


import struct


def pack_unpack(pack, unpack, v):

    if isinstance(v, list):
        return [struct.unpack(unpack, struct.pack(pack, _))[0] for _ in v]

    return struct.unpack(unpack, struct.pack(pack, v))[0]
