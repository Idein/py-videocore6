
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np
import itertools

def rotate_right(n, s):
    return ((n << (32-s)) | (n >> s)) & 0xffffffff

def count_leading_zeros(n):
    bit = 0x80000000
    count = 0
    while bit != n & bit:
        count += 1
        bit >>= 1
    return count

ops = {
    # binary ops
    'fadd' : lambda a,b: a + b,
    'faddnf' : lambda a,b: a + b,
    'fsub' : lambda a,b: a - b,
    'fmin' : np.minimum,
    'fmax' : np.maximum,
    'fmul' : lambda a,b: a * b,
    'fcmp' : lambda a,b: a - b,
    'vfpack' : lambda a,b: np.stack([a,b]).T.ravel(),
    'vfmin' : np.minimum,
    'vfmax' : np.maximum,
    'vfmul' : lambda a,b: a * b,

    'add' : lambda a,b: a + b,
    'sub' : lambda a,b: a - b,
    'imin' : np.minimum,
    'imax' : np.maximum,
    'umin' : np.minimum,
    'umax' : np.maximum,

    'shl' : lambda a,b: a << b,
    'shr' : lambda a,b: a >> b,
    'asr' : lambda a,b: a.astype('int32') >> b,
    'ror' : np.vectorize(rotate_right),

    'band' : lambda a,b: a & b,
    'bor' : lambda a,b: a | b,
    'bxor' : lambda a,b: a ^ b,

    # unary ops
    'fmov' : lambda x: x,
    'fround' : np.round,
    'ftrunc' : np.trunc,
    'ffloor' : np.floor,
    'fceil' : np.ceil,
    'fdx' : lambda x: (x[1::2] - x[0::2]).repeat(2),
    'fdy' : lambda x: (lambda a: (a[1::2] - a[0::2]).ravel())(x.reshape(-1,2).repeat(2,axis=0).reshape(-1,4)),
    'ftoin': lambda x: x.round().astype('int32'),
    'ftoiz': lambda x: np.trunc(x).astype('int32'),
    'ftouz': lambda x: np.trunc(x).astype('uint32'),

    'bnot' : lambda x: ~x,
    'neg' : lambda x: -x,

    'itof' : lambda x: x.astype('float32'),
    'clz' : np.vectorize(count_leading_zeros),
    'utof' : lambda x: x.astype('float32'),

    # pack/unpack flags
    'l' : lambda x: x[0::2],
    'h' : lambda x: x[1::2],
    None : lambda x: x,
    'none' : lambda x: x,
    'abs' : np.abs,
    'r32' : lambda x: x.repeat(2),
    'rl2h' : lambda x: x[0::2].repeat(2),
    'rh2l' : lambda x: x[1::2].repeat(2),
    'swap' : lambda x: x.reshape(-1,2)[:,::-1].ravel(),
}


@qpu
def qpu_binary_ops(asm, bin_ops, dst_ops, src1_ops, src2_ops):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif) # in
    mov(rf1, r5, sig = ldunif)  # out
    shl(r3, 4, 4).mov(rf2, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)
    add(rf2, rf2, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    mov(tmua, rf1, sig = thrsw).add(rf1, rf1, r3)
    nop(sig = ldtmu(r1))
    nop()
    nop(sig = ldtmu(r2))

    g = globals()
    for op, pack, unpack1, unpack2 in itertools.product(bin_ops, dst_ops, src1_ops, src2_ops):
        g[op](
            r0.pack(pack) if pack is not None else r0,
            r1.unpack(unpack1) if unpack1 is not None else r1,
            r2.unpack(unpack2) if unpack2 is not None else r2
        )
        mov(tmud, r0)
        mov(tmua, rf2)
        tmuwt().add(rf2, rf2, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def boilerplate_binary_ops(bin_ops, dst, src1, src2):

    dst_dtype, dst_ops = dst
    src1_dtype, src1_ops = src1
    src2_dtype, src2_ops = src2

    with Driver() as drv:

        cases = list(itertools.product(bin_ops, dst_ops, src1_ops, src2_ops))

        code = drv.program(lambda asm: qpu_binary_ops(asm, bin_ops, dst_ops, src1_ops, src2_ops))
        X1 = drv.alloc((16*4//np.dtype(src1_dtype).itemsize, ), dtype = src1_dtype)
        X2 = drv.alloc((16*4//np.dtype(src2_dtype).itemsize, ), dtype = src2_dtype)
        Y = drv.alloc((len(cases), 16*4//np.dtype(dst_dtype).itemsize), dtype = dst_dtype)
        unif = drv.alloc(3, dtype = 'uint32')

        X1[:] = np.random.randn(*X1.shape).astype(src1_dtype)
        X2[:] = np.random.randn(*X2.shape).astype(src2_dtype)
        Y[:] = 0.0

        unif[0] = X1.addresses()[0]
        unif[1] = X2.addresses()[0]
        unif[2] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        for ix, (bin_op, dst_op, src1_op, src2_op) in enumerate(cases):
            msg = '{}({}, {}, {})'.format(bin_op, dst_op, src1_op, src2_op)
            if np.dtype(dst_dtype).name.startswith('float'):
                assert np.allclose(ops[dst_op](Y[ix]), ops[bin_op](ops[src1_op](X1), ops[src2_op](X2)), rtol=1e-2), msg
            elif np.dtype(dst_dtype).name.startswith('int') or np.dtype(dst_dtype).name.startswith('uint'):
                assert np.all(ops[dst_op](Y[ix]) == ops[bin_op](ops[src1_op](X1), ops[src2_op](X2))), msg

def test_binary_ops():
    packs = [('float32', [None, 'none']), ('float16', ['l', 'h'])]
    unpacks = [('float32', [None, 'none', 'abs']), ('float16', ['l', 'h'])]
    for dst, src1, src2 in itertools.product(packs, unpacks, unpacks):
        boilerplate_binary_ops(
            ['fadd', 'faddnf', 'fsub', 'fmin', 'fmax', 'fmul', 'fcmp'],
            dst, src1, src2,
        )
    packs = [('float16', [None, 'none'])]
    unpacks = [('float32', [None, 'none']), ('float16', ['l', 'h'])]
    for dst, src1, src2 in itertools.product(packs, unpacks, unpacks):
        boilerplate_binary_ops(
            ['vfpack'],
            dst, src1, src2,
        )
    packs = [('float16', [None, 'none'])]
    unpacks = [('float32', ['r32']), ('float16', ['rl2h', 'rh2l', 'swap'])]
    for dst, src1, src2 in itertools.product(packs, unpacks, packs):
        boilerplate_binary_ops(
            ['vfmin', 'vfmax', 'vfmul'],
            dst, src1, src2,
        )

    boilerplate_binary_ops(
        ['add', 'sub', 'imin', 'imax'],
        ('int32', [None]), ('int32', [None]), ('int32', [None]),
    )
    boilerplate_binary_ops(
        ['add', 'sub', 'umin', 'umax'],
        ('uint32', [None]), ('uint32', [None]), ('uint32', [None]),
    )
    boilerplate_binary_ops(
        ['shl', 'shr', 'asr', 'ror'],
        ('uint32', [None]), ('uint32', [None]), ('uint32', [None]),
    )
    boilerplate_binary_ops(
        ['band', 'bor', 'bxor'],
        ('uint32', [None]), ('uint32', [None]), ('uint32', [None]),
    )

@qpu
def qpu_unary_ops(asm, bin_ops, dst_ops, src_ops):

    eidx(r0, sig = ldunif)
    mov(rf0, r5, sig = ldunif) # in
    shl(r3, 4, 4).mov(rf1, r5)

    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw).add(rf0, rf0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r1))

    g = globals()
    for op, pack, unpack in itertools.product(bin_ops, dst_ops, src_ops):
        g[op](
            r0.pack(pack) if pack is not None else r0,
            r1.unpack(unpack) if unpack is not None else r1,
        )
        mov(tmud, r0)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, r3)

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def boilerplate_unary_ops(uni_ops, dst, src):

    dst_dtype, dst_ops = dst
    src_dtype, src_ops = src

    with Driver() as drv:

        cases = list(itertools.product(uni_ops, dst_ops, src_ops))

        code = drv.program(lambda asm: qpu_unary_ops(asm, uni_ops, dst_ops, src_ops))
        X = drv.alloc((16*4//np.dtype(src_dtype).itemsize, ), dtype = src_dtype)
        Y = drv.alloc((len(cases), 16*4//np.dtype(dst_dtype).itemsize), dtype = dst_dtype)
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.random.randn(*X.shape).astype(src_dtype)
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        for ix, (uni_op, dst_op, src_op) in enumerate(cases):
            msg = '{}({}, {})'.format(uni_op, dst_op, src_op)
            if np.dtype(dst_dtype).name.startswith('float'):
                assert np.allclose(ops[dst_op](Y[ix]), ops[uni_op](ops[src_op](X)), rtol=1e-2), msg
            elif np.dtype(dst_dtype).name.startswith('int') or np.dtype(dst_dtype).name.startswith('uint'):
                assert np.all(ops[dst_op](Y[ix]) == ops[uni_op](ops[src_op](X))), msg

def test_unary_ops():
    packs = [('float32', [None, 'none']), ('float16', ['l', 'h'])]
    unpacks = [('float32', [None, 'none', 'abs']), ('float16', ['l', 'h'])]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ['fmov'],
            dst, src,
        )
    packs = [('float32', [None, 'none']), ('float16', ['l', 'h'])]
    unpacks = [('float32', [None, 'none']), ('float16', ['l', 'h'])]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ['fround', 'ftrunc', 'ffloor', 'fceil', 'fdx', 'fdy'],
            dst, src,
        )
    packs = [('int32', [None, 'none'])]
    unpacks = [('float32', [None, 'none']), ('float16', ['l', 'h'])]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ['ftoin', 'ftoiz'],
            dst, src,
        )
    packs = [('uint32', [None, 'none'])]
    unpacks = [('float32', [None, 'none']), ('float16', ['l', 'h'])]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ['ftouz'],
            dst, src,
        )
    # TODO: 'ftoc': what is the meaning of this instruction ?
    # packs = [('int32', ['none'])]
    # unpacks = [('float32', ['none']), ('float16', ['l', 'h'])]
    # for dst, src in itertools.product(packs, unpacks):
    #     boilerplate_unary_ops(
    #         ['ftoc'],
    #         dst, src,
    #     )
    boilerplate_unary_ops(
        ['bnot', 'neg'],
        ('int32', [None]), ('int32', [None]),
    )
    boilerplate_unary_ops(
        ['itof'],
        ('float32', [None]), ('int32', [None]),
    )
    boilerplate_unary_ops(
        ['clz'],
        ('uint32', [None]), ('uint32', [None]),
    )
    boilerplate_unary_ops(
        ['utof'],
        ('float32', [None]), ('uint32', [None]),
    )
