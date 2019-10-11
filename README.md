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

## Installation

```
$ git clone https://github.com/Idein/py-videocore6.git
$ cd py-videocore6/
$ pip3 install --user --requirement requirements.txt
$ python3 setup.py build
$ python3 setup.py install --user
```

## Testing

```
$ python3 setup.py build_ext --inplace
$ pip3 install nose
$ nosetests -v -s
```
