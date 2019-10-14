# py-videocore6

Python library for GPGPU programming on Raspberry Pi 4.

The V3D DRM driver is based on
[linux/drivers/gpu/drm/v3d](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/drivers/gpu/drm/v3d)
and the assembler is based on
[mesa/src/broadcom/qpu](https://gitlab.freedesktop.org/mesa/mesa/tree/master/src/broadcom/qpu).
Especially, the disassembler in the Mesa repository is a great help in
reverse-engineering the instruction set of VideoCore VI QPU.
You can try it with
[Terminus-IMRC/vc6qpudisas](https://github.com/Terminus-IMRC/vc6qpudisas).

For Raspberry Pi 1/2/3, use
[nineties/py-videocore](https://github.com/nineties/py-videocore) instead.

## About VideoCore VI QPU

Raspberry Pi 4 has a GPU named VideoCore VI QPU in its SoC.
Though the basic instruction set (add/mul ALU dual issue, three delay slots
et al.) remains same as VideoCore IV QPU of Raspberry Pi 1/2/3,
the usages of some units are dramatically changed.
For instance, the TMU can now write to memory in addition to read.
Consult the [tests](./tests) directory for more examples.

Theoretical peak performances of QPUs are as follows.
Note that the V3D DRM driver does not seem to support multi-sliced CSD job for
now.

- VideoCore IV QPU @ 250MHz: 250 [MHz] x 3 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 24 [Gflop/s]
- VideoCore IV QPU @ 300MHz: 300 [MHz] x 3 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 28.8 [Gflop/s]
- VideoCore VI QPU @ 500MHz: 500 [MHz] x 2 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 32 [Gflop/s]

## Installation

```
$ apt-get update
$ apt-get install python3-pip
$ pip3 install --user git+https://github.com/Idein/py-videocore6.git
```

## Testing

```
$ pip3 install --user nose
$ git clone https://github.com/Idein/py-videocore6.git
$ cd py-videocore6/
$ nosetests -v -s
```
