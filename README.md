# py-videocore6

Python library for GPGPU programming on Raspberry Pi 4.

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
