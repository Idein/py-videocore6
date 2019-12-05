
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np

def sfu_sin(x):
    result = np.sin(x * np.pi)
    result[x < -0.5] = -1
    result[x >  0.5] = 1
    return result

ops = {
    # sfu ops
    'op_recip' : lambda x: 1 / x,
    'op_rsqrt' : lambda x: 1 / np.sqrt(x),
    'op_exp' : lambda x: 2 ** x,
    'op_log' : np.log2,
    'op_sin' : sfu_sin,
    'op_rsqrt2' : lambda x: 1 / np.sqrt(x),
}



# SFU IO registers
@qpu
def qpu_sfu_regs(asm, sfu_regs):

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
    for reg in sfu_regs:
        mov(g[reg], r1)
        nop(null) # required ? enough ?
        mov(tmud, r4)
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

def boilerplate_sfu_regs(sfu_regs, domain_limitter):

    with Driver() as drv:

        code = drv.program(lambda asm: qpu_sfu_regs(asm, sfu_regs))
        X = drv.alloc((16, ), dtype = 'float32')
        Y = drv.alloc((len(sfu_regs), 16), dtype = 'float32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = domain_limitter(np.random.randn(*X.shape).astype('float32'))
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        for ix, reg in enumerate(sfu_regs):
            msg = 'mov({}, None)'.format(reg)
            assert np.allclose(Y[ix], ops['op_'+reg](X), rtol=1e-4), msg

def test_sfu_regs():
    boilerplate_sfu_regs(['recip','exp','sin'], lambda x: x)
    boilerplate_sfu_regs(['rsqrt','log','rsqrt2'], lambda x: x ** 2 + 1e-6)


# SFU ops
@qpu
def qpu_sfu_ops(asm, sfu_ops):

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
    for op in sfu_ops:
        g['op_'+op](rf2, r1) # ATTENTION: SFU ops requires rfN ?
        nop(null)
        mov(tmud, rf2)
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

def boilerplate_sfu_ops(sfu_ops, domain_limitter):

    with Driver() as drv:

        code = drv.program(lambda asm: qpu_sfu_ops(asm, sfu_ops))
        X = drv.alloc((16, ), dtype = 'float32')
        Y = drv.alloc((len(sfu_ops), 16), dtype = 'float32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = domain_limitter(np.random.randn(*X.shape).astype('float32'))
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        for ix, op in enumerate(sfu_ops):
            msg = '{}(None, None)'.format(op)
            assert np.allclose(Y[ix], ops['op_'+op](X), rtol=1e-4), msg

def test_sfu_ops():
    boilerplate_sfu_ops(['recip','exp','sin'], lambda x: x)
    boilerplate_sfu_ops(['rsqrt','log','rsqrt2'], lambda x: x ** 2 + 1e-6)
