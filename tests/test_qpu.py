
import time
from videocore6.driver import Driver
from videocore6.assembler import qpu


@qpu
def qpu_clock(asm):

    nop(null, sig = 'ldunif')

    L.loop
    sub(r5, r5, 1, cond = 'pushn')
    b(R.loop, cond = 'anyna')
    nop(null)
    nop(null)
    nop(null)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)


def test_clock():
    print()

    with Driver() as drv:

        f = pow(2, 25)

        code = drv.program(qpu_clock)
        unif = drv.alloc(1, dtype = 'uint32')

        unif[0] = f

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        print(f'{end - start} sec')
        print(f'{f * 5 / (end - start) / 1000 / 1000 * 4} MHz')


@qpu
def qpu_tmu_write(asm):

    nop(null, sig = 'ldunif')
    bor(r1, r5, r5, sig = 'ldunif')

    eidx(rf0)

    # r5 += eidx * 4
    eidx(r0)
    shl(r0, r0, 2)
    add(r5, r5, r0)

    L.loop
    if True:

        # rf0: Data to be written.
        # r0: Overwritten.
        # r5: Address to write data to.

        # r0 = eidx & 0b11
        eidx(r0)
        band(r0, r0, 0b11)

        # TMU writes just four elements at once without uniforms (it seems).
        for i in range(4):
            bor(tmud, rf0, rf0).sub(null, r0, i, cond = 'pushz')
            add(tmua, r5, i * 4, cond = 'ifa')

        tmuwt(null)

        sub(r1, r1, 1, cond = 'pushz')
        b(R.loop, cond = 'anyna')
        # rf0 += 16
        sub(rf0, rf0, -16)
        # r5 += 64
        shl(r0, 4, 4)
        add(r5, r5, r0)

    nop(null, sig = 'thrsw')
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null, sig = 'thrsw')
    nop(null)
    nop(null)
    nop(null)


def test_tmu_write():
    print()

    n = 256 * 1024

    with Driver(data_area_size = n * 16 * 4 + 2 * 4) as drv:

        code = drv.program(qpu_tmu_write)
        unif = drv.alloc(2, dtype = 'uint32')
        data = drv.alloc(n * 16, dtype = 'uint32')

        data[:] = 0xdeadbeaf
        unif[0] = n
        unif[1] = data.addresses()[0]

        start = time.time()
        drv.execute(code, unif.addresses()[0])
        end = time.time()

        assert all(data == range(n * 16))

        print(f'{end - start} sec')
        print(f'{data.nbytes / (end - start) / 1000 / 1000} MB/s')
