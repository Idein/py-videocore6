from typing import Final

from _videocore6.assembler import ALU as ALU
from _videocore6.assembler import Assembly as Assembly
from _videocore6.assembler import Branch as Branch
from _videocore6.assembler import Label as Label
from _videocore6.assembler import LabelNameSpace as LabelNameSpace
from _videocore6.assembler import Link as Link
from _videocore6.assembler import LoopHelper as LoopHelper
from _videocore6.assembler import Reference as Reference
from _videocore6.assembler import Register as Register
from _videocore6.assembler import RotateSignal as RotateSignal
from _videocore6.assembler import SFUIntegrator as SFUIntegrator
from _videocore6.assembler import Signal as Signal
from _videocore6.assembler import SignalArg as SignalArg
from _videocore6.assembler import WriteSignal as WriteSignal
from _videocore6.assembler import qpu as qpu

# Structured programming helpers
loop: Final[LoopHelper]
L: Final[Label]
R: Final[Reference]

def b(
    src: int | Register | Reference | Link | None,
    *,
    cond: str,
    absolute: bool = False,
    set_link: bool = False,
) -> Branch: ...

link: Final[Link]

def namespace(name: str) -> LabelNameSpace: ...

# Signals
thrsw: Final[Signal]
ldunif: Final[Signal]
ldunifa: Final[Signal]
ldunifrf: Final[WriteSignal]
ldunifarf: Final[WriteSignal]
ldtmu: Final[WriteSignal]
ldvary: Final[WriteSignal]
ldvpm: Final[Signal]
ldtlb: Final[WriteSignal]
ldtlbu: Final[WriteSignal]
smimm: Final[Signal]
ucb: Final[Signal]
rot: Final[RotateSignal]
wrtmuc: Final[Signal]

# Registers
r0: Final[Register]
r1: Final[Register]
r2: Final[Register]
r3: Final[Register]
r4: Final[Register]
r5: Final[Register]
null: Final[Register]
tlb: Final[Register]
tlbu: Final[Register]
unifa: Final[Register]
tmul: Final[Register]
tmud: Final[Register]
tmua: Final[Register]
tmuau: Final[Register]
vpm: Final[Register]
vpmu: Final[Register]
sync: Final[Register]
syncu: Final[Register]
syncb: Final[Register]
recip: Final[SFUIntegrator]
rsqrt: Final[SFUIntegrator]
exp: Final[SFUIntegrator]
log: Final[SFUIntegrator]
sin: Final[SFUIntegrator]
rsqrt2: Final[SFUIntegrator]
tmuc: Final[Register]
tmus: Final[Register]
tmut: Final[Register]
tmur: Final[Register]
tmui: Final[Register]
tmub: Final[Register]
tmudref: Final[Register]
tmuoff: Final[Register]
tmuscm: Final[Register]
tmusf: Final[Register]
tmuslod: Final[Register]
tmuhs: Final[Register]
tmuhscm: Final[Register]
tmuhsf: Final[Register]
tmuhslod: Final[Register]
r5rep: Final[Register]

# Register Alias
broadcast: Final[Register]  # r5rep
quad_broadcast: Final[Register]  # r5

# Register Files
rf: Final[list[Register]]
rf0: Final[Register]
rf1: Final[Register]
rf2: Final[Register]
rf3: Final[Register]
rf4: Final[Register]
rf5: Final[Register]
rf6: Final[Register]
rf7: Final[Register]
rf8: Final[Register]
rf9: Final[Register]
rf10: Final[Register]
rf11: Final[Register]
rf12: Final[Register]
rf13: Final[Register]
rf14: Final[Register]
rf15: Final[Register]
rf16: Final[Register]
rf17: Final[Register]
rf18: Final[Register]
rf19: Final[Register]
rf20: Final[Register]
rf21: Final[Register]
rf22: Final[Register]
rf23: Final[Register]
rf24: Final[Register]
rf25: Final[Register]
rf26: Final[Register]
rf27: Final[Register]
rf28: Final[Register]
rf29: Final[Register]
rf30: Final[Register]
rf31: Final[Register]
rf32: Final[Register]
rf33: Final[Register]
rf34: Final[Register]
rf35: Final[Register]
rf36: Final[Register]
rf37: Final[Register]
rf38: Final[Register]
rf39: Final[Register]
rf40: Final[Register]
rf41: Final[Register]
rf42: Final[Register]
rf43: Final[Register]
rf44: Final[Register]
rf45: Final[Register]
rf46: Final[Register]
rf47: Final[Register]
rf48: Final[Register]
rf49: Final[Register]
rf50: Final[Register]
rf51: Final[Register]
rf52: Final[Register]
rf53: Final[Register]
rf54: Final[Register]
rf55: Final[Register]
rf56: Final[Register]
rf57: Final[Register]
rf58: Final[Register]
rf59: Final[Register]
rf60: Final[Register]
rf61: Final[Register]
rf62: Final[Register]
rf63: Final[Register]

# Add ALU instructions
def fadd(
    dst: Register, src1: Register | float, src2: Register | float, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def faddnf(
    dst: Register, src1: Register | float, src2: float | float, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def vfpack(
    dst: Register, src1: Register | float, src2: Register | float, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def add(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def sub(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def fsub(
    dst: Register, src1: Register | float, src2: Register | float, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def imin(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def imax(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def umin(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def umax(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def shl(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def shr(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def asr(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def ror(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def fmin(
    dst: Register, src1: Register | float, src2: Register | float, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def fmax(
    dst: Register, src1: Register | float, src2: Register | float, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def vfmin(
    dst: Register, src1: Register | float, src2: Register | float, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def band(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def bor(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def bxor(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def bnot(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def neg(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def flapush(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def flbpush(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def flpop(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def setmsf(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def setrevf(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def nop(cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def tidx(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def eidx(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def lr(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def vfla(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def vflna(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def vflb(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def vflnb(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def msf(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def revf(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def iid(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def sampid(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def barrierid(dst: Register, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def tmuwt(dst: Register = null, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def vpmwt(dst: Register = null, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def fcmp(
    dst: Register, src1: Register | float, src2: Register | float, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def vfmax(
    dst: Register,
    src1: Register | int | float,
    src2: Register | int | float,
    cond: str | None = None,
    sig: SignalArg = None,
) -> ALU: ...

#
def fround(dst: Register, src: Register | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def ftoin(dst: Register, src: Register | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def ftrunc(dst: Register, src: Register | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def ftoiz(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def ffloor(dst: Register, src: Register | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def ftouz(dst: Register, src: Register | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def fceil(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def ftoc(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def fdx(dst: Register, src: Register | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def fdy(dst: Register, src: Register | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def stvpmv(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def stvpmd(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def stvpmp(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def itof(dst: Register, src: Register | int, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def clz(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def utof(dst: Register, src: Register | int, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

# Mul ALU instructions
def umul24(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def vfmul(
    dst: Register, src1: Register | float, src2: Register | float, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def smul24(
    dst: Register, src1: Register | int, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def multop(
    dst: Register, src1: Register | int, src2: Register | float, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def fmov(dst: Register, src: Register | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def mov(dst: Register, src: Register | int | float, cond: str | None = None, sig: SignalArg = None) -> ALU: ...

#
def fmul(
    dst: Register, src1: Register | float, src2: Register | float, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def quad_rotate(
    dst: Register, src1: Register | int | float, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...

#
def rotate(
    dst: Register, src1: Register | int | float, src2: Register | int, cond: str | None = None, sig: SignalArg = None
) -> ALU: ...
