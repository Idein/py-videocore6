
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu
import numpy as np


@qpu
def qpu_pack(asm):

    eidx(r0, sig = 'ldunif')
    mov(r2, r5)
    shl(r0, r0, 2)
    add(r2, r2, r0)
    shl(r1, 4, 4)

    eidx(r3)
    itof(r3, r3)
    fadd(r0.pack('h'), r3, r3)
    fadd(r0.pack('l'), r3, 0)
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    eidx(r3)
    itof(r3, r3)
    faddnf(r0.pack('h'), r3, r3)
    faddnf(r0.pack('l'), r3, 0)
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    eidx(r3)
    itof(r3, r3)
    fsub(r0.pack('h'), r3, r3)
    fsub(r0.pack('l'), 0, r3)
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    eidx(r3)
    itof(r3, r3)
    fmin(r0.pack('h'), r3, 4.0)
    fmin(r0.pack('l'), 8.0, r3)
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    eidx(r3)
    itof(r3, r3)
    fmax(r0.pack('h'), r3, 4.0)
    fmax(r0.pack('l'), 8.0, r3)
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)

def test_pack():

    with Driver() as drv:

        code = drv.program(qpu_pack)
        data = drv.alloc((5, 32), dtype = 'float16')
        unif = drv.alloc(1, dtype = 'uint32')

        data[:] = 0

        unif[0] = data.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.arange(16, dtype='float16')
        assert (data[0,0:32:2] == expected).all()
        assert (data[0,1:32:2] == (expected * 2)).all()
        assert (data[1,0:32:2] == expected).all()
        assert (data[1,1:32:2] == (expected * 2)).all()
        assert (data[2,0:32:2] == -expected).all()
        assert (data[2,1:32:2] == 0).all()
        assert (data[3,0:32:2] == np.minimum(expected, 8)).all()
        assert (data[3,1:32:2] == np.minimum(expected, 4)).all()
        assert (data[4,0:32:2] == np.maximum(expected, 8)).all()
        assert (data[4,1:32:2] == np.maximum(expected, 4)).all()


@qpu
def qpu_unpack_lr(asm):

    eidx(r0, sig = 'ldunif')
    mov(r2, r5)
    shl(r0, r0, 2)
    add(r2, r2, r0)
    shl(r1, 4, 4)

    eidx(r3).mov(r0, 0)
    itof(r0, r3)
    fadd(r3.pack('l'), 0, r0)
    fsub(r3.pack('h'), 0, r0)

    fadd(r0, r3.unpack('l'), r3.unpack('l'))
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    fadd(r0, r3.unpack('l'), r3.unpack('h'))
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    fadd(r0, r3.unpack('h'), r3.unpack('l'))
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    fadd(r0, r3.unpack('h'), r3.unpack('h'))
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    fsub(r0, r3.unpack('l'), r3.unpack('l'))
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    fsub(r0, r3.unpack('l'), r3.unpack('h'))
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    fsub(r0, r3.unpack('h'), r3.unpack('l'))
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)
    fsub(r0, r3.unpack('h'), r3.unpack('h'))
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)

def test_unpack_lr():

    with Driver() as drv:

        code = drv.program(qpu_unpack_lr)
        data = drv.alloc((8, 16), dtype = 'float32')
        unif = drv.alloc(1, dtype = 'uint32')

        data[:] = 0

        unif[0] = data.addresses()[0,0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        l = np.arange(16, dtype='float16')
        h = - np.arange(16, dtype='float16')
        assert (data[0] == l + l).all()
        assert (data[1] == l + h).all()
        assert (data[2] == h + l).all()
        assert (data[3] == h + h).all()
        assert (data[4] == l - l).all()
        assert (data[5] == l - h).all()
        assert (data[6] == h - l).all()
        assert (data[7] == h - h).all()

@qpu
def qpu_unpack_abs(asm):

    eidx(r0, sig = 'ldunif')
    mov(r2, r5)
    shl(r0, r0, 2)
    add(r2, r2, r0)
    shl(r1, 4, 4)

    eidx(r3)
    sub(r3, r3, 10)
    itof(r3, r3)
    fadd(r0, r3.unpack('abs'), 0)
    mov(tmud, r0)
    mov(tmua, r2)
    tmuwt(null).add(r2, r2, r1)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)

def test_unpack_abs():

    with Driver() as drv:

        code = drv.program(qpu_unpack_abs)
        data = drv.alloc((16, ), dtype = 'float32')
        unif = drv.alloc(1, dtype = 'uint32')

        data[:] = 0

        unif[0] = data.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        expected = np.abs(np.arange(16, dtype='float16') - 10)
        assert (data == expected).all()
