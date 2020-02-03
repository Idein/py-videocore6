
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np

@qpu
def qpu_label_with_namespace(asm):

    mov(r0, 0)

    with namespace('ns1'):
        b(R.test, cond = 'always')
        nop()
        nop()
        nop()
        add(r0, r0, 10)
        L.test
        add(r0, r0, 1)

        with namespace('nested'):
            b(R.test, cond = 'always')
            nop()
            nop()
            nop()
            add(r0, r0, 10)
            L.test
            add(r0, r0, 1)

    with namespace('ns2'):
        b(R.test, cond = 'always')
        nop()
        nop()
        nop()
        add(r0, r0, 10)
        L.test
        add(r0, r0, 1)

    b(R.test, cond = 'always')
    nop()
    nop()
    nop()
    add(r0, r0, 10)
    L.test
    add(r0, r0, 1)

    with namespace('ns3'):
        b(R.test, cond = 'always')
        nop()
        nop()
        nop()
        add(r0, r0, 10)
        L.test
        add(r0, r0, 1)

    eidx(r1, sig = ldunifrf(rf2))
    shl(r1, r1, 2)

    mov(tmud, r0)
    add(tmua, rf2, r1)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_label_with_namespace():

    with Driver() as drv:

        code = drv.program(qpu_label_with_namespace)
        data = drv.alloc(16, dtype = 'uint32')
        unif = drv.alloc(1, dtype = 'uint32')

        data[:] = 1234

        unif[0] = data.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert (data == 5).all()
