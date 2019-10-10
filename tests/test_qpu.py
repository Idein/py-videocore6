
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
        drv.execute(code, unif.addresses())
        end = time.time()

        print(f'{end - start} sec')
        print(f'{f * 5 / (end - start) / 1000 / 1000 * 4} MHz')
