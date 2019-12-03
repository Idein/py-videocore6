
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np
import itertools


@qpu
def qpu_pack_unpack_binary_ops(asm, bin_ops, dst_ops, src1_ops, src2_ops):

    eidx(r0, sig = 'ldunif')
    mov(rf0, r5, sig = 'ldunif') # in
    mov(rf1, r5, sig = 'ldunif')  # out
    shl(r3, 4, 4).mov(rf2, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)
    add(rf2, rf2, r0)

    mov(tmua, rf0, sig = 'thrsw').add(rf0, rf0, r3)
    nop(null)
    mov(tmua, rf1, sig = 'thrsw').add(rf1, rf1, r3)
    nop(r1, sig = 'ldtmu')
    nop(null)
    nop(r2, sig = 'ldtmu')

    g = globals()
    for op, pack, unpack1, unpack2 in itertools.product(bin_ops, dst_ops, src1_ops, src2_ops):
        g[op](r0.pack(pack), r1.unpack(unpack1), r2.unpack(unpack2))
        mov(tmud, r0)
        mov(tmua, rf2)
        tmuwt(null).add(rf2, rf2, r3)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)

def boilerplate_pack_unpack_binary_ops(bin_ops, dst, src1, src2):

    dst_bits, dst_ops = dst
    src1_bits, src1_ops = src1
    src2_bits, src2_ops = src2

    ops = {
        # op
        'fadd' : lambda a,b: a + b,
        'faddnf' : lambda a,b: a + b,
        'fsub' : lambda a,b: a - b,
        'fmin' : np.minimum,
        'fmax' : np.maximum,
        'fmul' : lambda a,b: a * b,
        'vfpack' : lambda a,b: np.stack([a,b]).T.ravel(),
        'vfmin' : np.minimum,
        'vfmax' : np.maximum,
        'vfmul' : lambda a,b: a * b,
        # pack/unpack flags
        'l' : lambda x: x[0::2],
        'h' : lambda x: x[1::2],
        'none' : lambda x: x,
        'abs' : np.abs,
        'r32' : lambda x: np.stack([x,x]).T.ravel(),
        'rl2h' : lambda x: np.stack([x[0::2],x[0::2]]).T.ravel(),
        'rh2l' : lambda x: np.stack([x[1::2],x[1::2]]).T.ravel(),
        'swap' : lambda x: np.stack([x[1::2],x[0::2]]).T.ravel(),
    }

    with Driver() as drv:

        cases = list(itertools.product(bin_ops, dst_ops, src1_ops, src2_ops))

        code = drv.program(lambda asm: qpu_pack_unpack_binary_ops(asm, bin_ops, dst_ops, src1_ops, src2_ops))
        X1 = drv.alloc((48-src1_bits, ), dtype = 'float{}'.format(src1_bits))
        X2 = drv.alloc((48-src2_bits, ), dtype = 'float{}'.format(src2_bits))
        Y = drv.alloc((len(cases), 48-dst_bits), dtype = 'float{}'.format(dst_bits))
        unif = drv.alloc(3, dtype = 'uint32')

        X1[:] = np.random.randn(*X1.shape).astype('float32')
        X2[:] = np.random.randn(*X2.shape).astype('float32')
        Y[:] = 0.0

        unif[0] = X1.addresses()[0]
        unif[1] = X2.addresses()[0]
        unif[2] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        for ix, (bin_op, dst_op, src1_op, src2_op) in enumerate(cases):
            msg = '{}({}, {}, {})'.format(bin_op, dst_op, src1_op, src2_op)
            assert np.allclose(ops[dst_op](Y[ix]), ops[bin_op](ops[src1_op](X1), ops[src2_op](X2)), rtol=1e-2), msg

def test_pack_unpack_binary_ops():
    packs = [(32, ['none']), (16, ['l', 'h'])]
    unpacks = [(32, ['none', 'abs']), (16, ['l', 'h'])]
    for dst, src1, src2 in itertools.product(packs, unpacks, unpacks):
        boilerplate_pack_unpack_binary_ops(
            ['fadd', 'faddnf', 'fsub', 'fmin', 'fmax', 'fmul'],
            dst, src1, src2,
        )
    packs = [(16, ['none'])]
    unpacks = [(32, ['none']), (16, ['l', 'h'])]
    for dst, src1, src2 in itertools.product(packs, unpacks, unpacks):
        boilerplate_pack_unpack_binary_ops(
            ['vfpack'],
            dst, src1, src2,
        )
    packs = [(16, ['none'])]
    unpacks = [(32, ['r32']), (16, ['rl2h', 'rh2l', 'swap'])]
    for dst, src1, src2 in itertools.product(packs, unpacks, packs):
        boilerplate_pack_unpack_binary_ops(
            ['vfmin', 'vfmax', 'vfmul'],
            dst, src1, src2,
        )

@qpu
def qpu_pack_unpack_unary_ops(asm, bin_ops, dst_ops, src_ops):

    eidx(r0, sig = 'ldunif')
    mov(rf0, r5, sig = 'ldunif') # in
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = 'thrsw').add(rf0, rf0, r3)
    nop(null)
    nop(null)
    nop(r1, sig = 'ldtmu')

    g = globals()
    for op, pack, unpack in itertools.product(bin_ops, dst_ops, src_ops):
        g[op](r0.pack(pack), r1.unpack(unpack))
        mov(tmud, r0)
        mov(tmua, rf1)
        tmuwt(null).add(rf1, rf1, r3)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)

def boilerplate_pack_unpack_unary_ops(uni_ops, dst, src):

    dst_bits, dst_ops = dst
    src_bits, src_ops = src

    ops = {
        # op
        'fmov' : lambda x: x,
        # pack/unpack flags
        'l' : lambda x: x[0::2],
        'h' : lambda x: x[1::2],
        'none' : lambda x: x,
        'abs' : np.abs,
    }

    with Driver() as drv:

        cases = list(itertools.product(uni_ops, dst_ops, src_ops))

        code = drv.program(lambda asm: qpu_pack_unpack_unary_ops(asm, uni_ops, dst_ops, src_ops))
        X = drv.alloc((48-src_bits, ), dtype = 'float{}'.format(src_bits))
        Y = drv.alloc((len(cases), 48-dst_bits), dtype = 'float{}'.format(dst_bits))
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.random.randn(*X.shape).astype('float32')
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        for ix, (uni_op, dst_op, src_op) in enumerate(cases):
            msg = '{}({}, {})'.format(uni_op, dst_op, src_op)
            assert np.allclose(ops[dst_op](Y[ix]), ops[uni_op](ops[src_op](X)), rtol=1e-2), msg

def test_pack_unpack_unary_ops():
    packs = [(32, ['none']), (16, ['l', 'h'])]
    unpacks = [(32, ['none', 'abs']), (16, ['l', 'h'])]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_pack_unpack_unary_ops(
            ['fmov'],
            dst, src,
        )
