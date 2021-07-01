# py-videocore6

A Python library for GPGPU programming on Raspberry Pi 4, which realizes
assembling and running QPU programs.

For Raspberry Pi Zero/1/2/3, use
[nineties/py-videocore](https://github.com/nineties/py-videocore) instead.


## About VideoCore VI QPU

Raspberry Pi 4 (BCM2711) has a GPU named VideoCore VI QPU in its SoC.
The basic instruction set (add/mul ALU dual issue, three delay slots et al.)
remains the same as VideoCore IV QPU of Raspberry Pi Zero/1/2/3, and some units
now perform differently.
For instance, the TMU can now write to memory in addition to read, and it seems
that the VPM DMA is no longer available.

Theoretical peak performance of QPUs are as follows.

- VideoCore IV QPU @ 250MHz: 250 [MHz] x 3 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 24 [Gflop/s]
- VideoCore IV QPU @ 300MHz: 300 [MHz] x 3 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 28.8 [Gflop/s]
- VideoCore VI QPU @ 500MHz: 500 [MHz] x 2 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 32 [Gflop/s]


## Requirements

`py-videocore6` communicates with the V3D hardware through `/dev/dri/card0`,
which is exposed by the DRM V3D driver.
To access the device, you need to belong to `video` group or be `root` user.
If you choose the former, run `sudo usermod --append --groups video $USER`
(re-login to take effect).


## Installation

You can install `py-videocore6` directly using `pip`:

```console
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install python3-pip python3-numpy
$ pip3 install --user --upgrade pip setuptools wheel
$ pip3 install --user git+https://github.com/Idein/py-videocore6.git
```

If you are willing to run tests and examples, install `py-videocore6` after
cloning it:

```console
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install python3-pip python3-numpy libatlas3-base
$ python3 -m pip install --user --upgrade pip setuptools wheel
$ git clone https://github.com/Idein/py-videocore6.git
$ cd py-videocore6/
$ python3 -m pip install --target sandbox/ --upgrade . nose
```


## Running tests and examples

In the `py-videocore6` directory cloned above:

```console
$ python3 setup.py build_ext --inplace
$ PYTHONPATH=sandbox/ python3 -m nose -v -s
```

```console
$ PYTHONPATH=sandbox/ python3 examples/sgemm.py
==== sgemm example (1024x1024 times 1024x1024) ====
numpy: 0.6986 sec, 3.078 Gflop/s
QPU:   0.5546 sec, 3.878 Gflop/s
Minimum absolute error: 0.0
Maximum absolute error: 0.0003814697265625
Minimum relative error: 0.0
Maximum relative error: 0.13375753164291382
```

```console
$ PYTHONPATH=sandbox/ python3 examples/summation.py
==== summaton example (32.0 Mi elements) ====
Preparing for buffers...
Executing on QPU...
0.01853448400004254 sec, 7241.514141947083 MB/s
```

```console
$ PYTHONPATH=sandbox/ python3 examples/memset.py
==== memset example (64.0 MiB) ====
Preparing for buffers...
Executing on QPU...
0.01788834699993913 sec, 3751.5408215319367 MB/s
```

```console
$ PYTHONPATH=sandbox/ python3 examples/scopy.py
==== scopy example (16.0 Mi elements) ====
Preparing for buffers...
Executing on QPU...
0.02768789600000332 sec, 2423.761776625857 MB/s
```

```console
$ sudo PYTHONPATH=sandbox/ python3 examples/pctr_gpu_clock.py
==== QPU clock measurement with performance counters ====
500.529835 MHz
```

You may see lower performance without `force_turbo=1` in `/boot/config.txt`.


## References

- DRM V3D driver which controls QPU via hardware V3D registers: [linux/drivers/gpu/drm/v3d](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/drivers/gpu/drm/v3d)
- Mesa library which partially includes the QPU instruction set: [mesa/src/broadcom/qpu](https://gitlab.freedesktop.org/mesa/mesa/tree/master/src/broadcom/qpu)
- Mesa also includes QPU program disassembler, which can be tested with: [Terminus-IMRC/vc6qpudisas](https://github.com/Terminus-IMRC/vc6qpudisas)
