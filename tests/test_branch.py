
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np

# relative jump (immediate value)
@qpu
def qpu_branch_rel_imm(asm):

    eidx(r0, sig = ldunifrf(rf0))
    nop(sig = ldunifrf(rf1))
    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r1))

    b(2*8, cond = 'always')
    nop()
    nop()
    nop()
    add(r1, r1, 1)
    add(r1, r1, 1)
    add(r1, r1, 1) # jump comes here
    add(r1, r1, 1)

    mov(tmud, r1)
    mov(tmua, rf1)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_branch_rel_imm():

    with Driver() as drv:

        code = drv.program(qpu_branch_rel_imm)
        X = drv.alloc((16, ), dtype = 'uint32')
        Y = drv.alloc((16, ), dtype = 'uint32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y == X + 2).all()

# relative jump (label)
@qpu
def qpu_branch_rel_label(asm):

    eidx(r0, sig = ldunifrf(rf0))
    nop(sig = ldunifrf(rf1))
    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r1))

    b(R.foo, cond = 'always')
    nop()
    nop()
    nop()
    add(r1, r1, 1)
    L.foo
    add(r1, r1, 1) # jump comes here
    L.bar
    add(r1, r1, 1)
    L.baz
    add(r1, r1, 1)

    mov(tmud, r1)
    mov(tmua, rf1)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_branch_rel_label():

    with Driver() as drv:

        code = drv.program(qpu_branch_rel_label)
        X = drv.alloc((16, ), dtype = 'uint32')
        Y = drv.alloc((16, ), dtype = 'uint32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = np.arange(16)
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y == X + 3).all()


# absolute jump (reg)
@qpu
def qpu_branch_abs_reg(asm):

    eidx(r0, sig = ldunifrf(rf0))
    nop(sig = ldunifrf(rf1))
    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)

    mov(tmua, rf0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(rf2))

    mov(r1, 0)
    b(rf2, cond = 'always')
    nop()
    nop()
    nop()
    L.label
    add(r1, r1, 1)
    add(r1, r1, 1)
    add(r1, r1, 1)
    add(r1, r1, 1) # jump comes here

    mov(tmud, r1)
    mov(tmua, rf1)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_branch_abs_reg():

    with Driver() as drv:

        code = drv.program(qpu_branch_abs_reg)
        X = drv.alloc((16, ), dtype = 'uint32')
        Y = drv.alloc((16, ), dtype = 'uint32')
        unif = drv.alloc(3, dtype = 'uint32')

        X[:] = code.addresses()[0] + 17*8
        Y[:] = 0.0

        unif[0] = X.addresses()[0]
        unif[1] = Y.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (Y == 1).all()
