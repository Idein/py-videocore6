
import time

from videocore6.v3d import *

with RegisterMapping() as regmap:

    with PerformanceCounter(regmap, [CORE_PCTR_CYCLE_COUNT]) as pctr:

        time.sleep(1)
        result = pctr.result()

print('==== QPU clock measurement with performance counters ====')
print(f'{result[0] * 1e-6} MHz')
