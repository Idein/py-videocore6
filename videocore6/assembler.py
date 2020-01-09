
import functools
from videocore6 import float_to_int, int_to_float, int_to_uint


class AssembleError(Exception):
    pass


class Assembly(list):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = {}
        self.label_name_spaces = []

    def _gen_unused_label(self, label_format='{}'):
        n = len(self.labels)
        label = label_format.format(n)
        while label in self.labels:
            n += 1
            next_label = label_format.format(n)
            assert label != next_label, 'Bug: Invalid label format'
            label = next_label

        return label_format.format(n)

    def _gen_ns_label_name(self, name):
        return '.'.join(self.label_name_spaces + [name])

class LabelNameSpace(object):
    'Label namespace controller.'

    def __init__(self, asm, name):
        super(LabelNameSpace, self).__init__()
        self.asm = asm
        self.name = name

    def __enter__(self):
        self.asm.label_name_spaces.append(self.name)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.asm.label_name_spaces.pop()


class Label(object):

    def __init__(self, asm):
        self.asm = asm

    def __getattr__(self, name):
        if name in self.asm.labels:
            raise AssembleError(f'Label is duplicated: {name}')
        self.asm.labels[self.asm._gen_ns_label_name(name)] = len(self.asm)


class Reference(object):

    def __init__(self, asm, name=None):
        self.asm = asm
        self.name = name

    def __getattr__(self, name):
        return Reference(self.asm, self.asm._gen_ns_label_name(name))

    def __int__(self):
        return self.asm.labels[self.name]


class Register(object):

    OUTPUT_MODIFIER = {
        # modifier: (f32, f16)
        'none': 0,
        'l': 1,
        'h': 2,
    }

    INPUT_MODIFIER = {
        # modifier: (f32, f16)
        'none': (1, 0),
        'abs': (0, None),
        'l': (2, None),
        'h': (3, None),
        'r32': (None, 1),
        'rl2h': (None, 2),
        'rh2l': (None, 3),
        'swap': (None, 4),
    }

    def __init__(self, name, magic, waddr, pack='none', unpack='none'):
        self.name = name
        self.magic = magic
        self.waddr = waddr
        self.pack_bits = Register.OUTPUT_MODIFIER[pack]
        self.unpack_bits = Register.INPUT_MODIFIER[unpack]

    def pack(self, modifier):
        if self.pack_bits != Register.OUTPUT_MODIFIER['none']:
            raise AssembleError('Conflict pack')
        return Register(self.name, self.magic, self.waddr, pack=modifier)

    def unpack(self, modifier):
        if self.unpack_bits != Register.INPUT_MODIFIER['none']:
            raise AssembleError('Conflict unpack')
        return Register(self.name, self.magic, self.waddr, unpack=modifier)


class Signal(object):

    def __init__(self, name, dst=None, rot=None):
        self.name = name
        self.dst = dst
        self.rot = rot

    def is_write(self):
        ''' Write (to destination register) signal '''
        return self.dst is not None

    def is_rotate(self):
        return self.rot is not None


class WriteSignal(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, dst):
        return Signal(self.name, dst=dst)


class RotateSignal(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, rot):
        return Signal(self.name, rot=rot)


class Signals(set):

    def __init__(self):
        super(Signals, self).__init__()

    def add(self, sigs):
        if isinstance(sigs, WriteSignal):
            sig = sigs
            raise AssembleError(f'"{sig}" requires destination register (ex. "{sig}(r0)")')
        if isinstance(sigs, RotateSignal):
            sig = sigs
            raise AssembleError(f'"{sig}" requires rotate number (ex. "{sig}(-1)")')
        elif isinstance(sigs, Signal):
            sig = sigs
            if sig.name in [s.name for s in self]:
                raise AssembleError(f'Signal "{sig}" is duplicated')
            if sig.rot == 0:  # ignore rot(0)
                return
            super(Signals, self).add(sig)
            if len([s.dst for s in self if s.dst is not None]) > 1:
                raise AssembleError('Too many signals that require destination register')
        elif isinstance(sigs, (list, tuple, set)):
            for sig in sigs:
                self.add(sig)
        else:
            raise AssembleError('Invalid signal object')

    def pack(self):
        return {
            frozenset(): 0,
            frozenset(['thrsw']): 1,
            frozenset(['ldunif']): 2,
            frozenset(['thrsw', 'ldunif']): 3,
            frozenset(['ldtmu']): 4,
            frozenset(['thrsw', 'ldtmu']): 5,
            frozenset(['ldtmu', 'ldunif']): 6,
            frozenset(['thrsw', 'ldtmu', 'ldunif']): 7,
            frozenset(['ldvary']): 8,
            frozenset(['thrsw', 'ldvary']): 9,
            frozenset(['ldvary', 'ldunif']): 10,
            frozenset(['thrsw', 'ldvary', 'ldunif']): 11,
            frozenset(['ldunifrf']): 12,
            frozenset(['thrsw', 'ldunifrf']): 13,
            frozenset(['smimm', 'ldvary']): 14,
            frozenset(['smimm']): 15,
            frozenset(['ldtlb']): 16,
            frozenset(['ldtlbu']): 17,
            frozenset(['wrtmuc']): 18,
            frozenset(['thrsw', 'wrtmuc']): 19,
            frozenset(['ldvary', 'wrtmuc']): 20,
            frozenset(['thrsw', 'ldvary', 'wrtmuc']): 21,
            frozenset(['ucb']): 22,
            frozenset(['rot']): 23,
            frozenset(['ldunifa']): 24,
            frozenset(['ldunifarf']): 25,
            frozenset(['smimm', 'ldtmu']): 31,
        }[frozenset([sig.name for sig in self])] << 53

    def is_write(self):
        return any([sig.is_write() for sig in self])

    def write_address(self):
        assert self.is_write()
        dst = [sig.dst for sig in self if sig.dst is not None][0]
        return (dst.magic << 6) | dst.waddr

    def is_rotate(self):
        return any([sig.is_rotate() for sig in self])

    def rotate_count(self):
        assert self.is_rotate()
        rot = [sig.rot for sig in self if sig.rot is not None][0]
        return rot


class Instruction(object):

    SIGNALS = {
        'thrsw': Signal('thrsw'),
        'ldunif': Signal('ldunif'),
        'ldunifa': Signal('ldunifa'),
        'ldunifrf': WriteSignal('ldunifrf'),
        'ldunifarf': WriteSignal('ldunifarf'),
        'ldtmu': WriteSignal('ldtmu'),
        'ldvary': WriteSignal('ldvary'),
        'ldvpm': Signal('ldvpm'),
        'ldtlb': WriteSignal('ldtlb'),
        'ldtlbu': WriteSignal('ldtlbu'),
        'smimm': Signal('smimm'),
        'ucb': Signal('ucb'),
        'rot': RotateSignal('rot'),
        'wrtmuc': Signal('wrtmuc'),
    }

    REGISTERS = {
        name: Register(name, 1, addr)
        for addr, name in enumerate([
            'r0',
            'r1',
            'r2',
            'r3',
            'r4',
            'r5',
            'null',
            'tlb',
            'tlbu',
            'tmu',
            'tmul',
            'tmud',
            'tmua',
            'tmuau',
            'vpm',
            'vpmu',
            'sync',
            'syncu',
            'syncb',
            '_reg_recip',
            '_reg_rsqrt',
            '_reg_exp',
            '_reg_log',
            '_reg_sin',
            '_reg_rsqrt2',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            'tmuc',
            'tmus',
            'tmut',
            'tmur',
            'tmui',
            'tmub',
            'tmudref',
            'tmuoff',
            'tmuscm',
            'tmusf',
            'tmuslod',
            'tmuhs',
            'tmuhscm',
            'tmuhsf',
            'tmuhslod',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            'r5rep',
        ])
    }
    for i in range(64):
        REGISTERS[f'rf{i}'] = Register(f'rf{i}', 0, i)

    def __init__(self, asm, *args, **kwargs):
        self.serial = len(asm)
        asm.append(self)

    def __int__(self):
        return self.pack()

    def pack(self):
        assert False, 'Bug: Not implemented'


class ALUConditions(object):

    def __init__(self, cond_add, cond_mul):
        self.cond_add = cond_add
        self.cond_mul = cond_mul

    def pack(self, sigs):

        if sigs.is_write():
            return sigs.write_address() << 46

        conds_push = {
            'pushz': 1,
            'pushn': 2,
            'pushc': 3,
        }
        conds_update = {
            'andz': 4,
            'andnz': 5,
            'nornz': 6,
            'norz': 7,
            'andn': 8,
            'andnn': 9,
            'nornn': 10,
            'norn': 11,
            'andc': 12,
            'andnc': 13,
            'nornc': 14,
            'norc': 15,
        }
        conds_insn = {
            'ifa': 0,
            'ifb': 1,
            'ifna': 2,
            'ifnb': 3,
        }

        add_insn = 1 * int(self.cond_add in conds_insn.keys())
        add_push = 2 * int(self.cond_add in conds_push.keys())
        add_update = 3 * int(self.cond_add in conds_update.keys())

        mul_insn = 1 * int(self.cond_mul in conds_insn.keys())
        mul_push = 2 * int(self.cond_mul in conds_push.keys())
        mul_update = 3 * int(self.cond_mul in conds_update.keys())

        add_cond = add_insn + add_push + add_update
        mul_cond = mul_insn + mul_push + mul_update

        result = [
            #     none | add_insn | add_push | add_update
            [0b0000000, 0b0100000, 0b0000000, 0b0000000],  # none
            [0b0110000, 0b1000000, 0b0110000, 0b1000000],  # mul_insn
            [0b0010000, 0b0100000,      None,      None],  # mul_push
            [0b0010000,      None,      None,      None],  # mul_update
        ][mul_cond][add_cond]

        if result is None:
            raise AssembleError(f'Conflict conditional flags "{self.cond_add}" and "{self.cond_mul}"')

        if add_push > 0:
            result |= conds_push[self.cond_add]
        if mul_push > 0:
            result |= conds_push[self.cond_mul]
        if add_update > 0:
            result |= conds_update[self.cond_add]
        if mul_update > 0:
            result |= conds_update[self.cond_mul]
        if mul_insn > 0:
            if add_insn > 0:
                result |= conds_insn[self.cond_mul] << 4
                result |= conds_insn[self.cond_add]
            elif add_update > 0:
                result |= conds_insn[self.cond_mul] << 4
            else:
                result |= conds_insn[self.cond_mul] << 2
        elif add_insn > 0:
            result |= conds_insn[self.cond_add] << 2

        return result << 46


class ALURaddrs(object):

    def __init__(self):
        self.a = None
        self.b = None

    def has_smimm(self):
        return isinstance(self.b, (int, float))

    def determine_mux(self, op_mux, regs):

        reg_a, reg_b = regs
        mux_a, mux_b = op_mux

        assert (mux_a is None and isinstance(reg_a, (int, float, Register))) or (mux_a is not None and reg_a is None)
        assert (mux_b is None and isinstance(reg_b, (int, float, Register))) or (mux_b is not None and reg_b is None)
        assert (mux_a is not None and mux_b is not None) or (mux_a is None)

        def _to_mux(reg):

            if isinstance(reg, (int, float)):
                assert reg == self.b, f'Bug: small immediate {reg} is not added yet'
                return 7
            elif isinstance(reg, Register):
                if reg.magic == 0:
                    if isinstance(self.a, Register) and self.a.name == reg.name:
                        return 6
                    if isinstance(self.b, Register) and self.b.name == reg.name:
                        return 7
                    assert reg == self.b, f'Bug: register {reg} is not added yet'
                elif reg.waddr < 6:
                    return reg.waddr
                else:
                    assert reg == self.b, f'Bug: register {reg} should be failed to added'
            else:
                assert False, 'Bug: Type error'

        if mux_a is None:
            mux_a = _to_mux(reg_a)
        if mux_b is None:
            mux_b = _to_mux(reg_b)

        return mux_a, mux_b

    def add(self, reg):

        if reg is None:
            return

        assert isinstance(reg, (int, float, Register, Signal)), 'Bug: Type error'

        if isinstance(reg, Signal):
            sig = reg
            if not sig.is_rotate():
                return
            if self.b is None:
                self.b = sig
            elif isinstance(self.b, Signal) and self.b.is_rotate() and self.b.rotate_count() == sig.rotate_count():
                pass
            else:
                if isinstance(self.b, (int, float)):
                    raise AssembleError(f'Conflict small immediates {self.b} and signal {reg}')
                elif isinstance(self.b, Register):
                    raise AssembleError(f'Conflict register {self.b} and signal {reg}')
                elif isinstance(self.b, Signal):
                    raise AssembleError(f'Conflict signal {self.b} and signal {reg}')
                else:
                    assert False, 'Bug: Lack of type exhaustivity'
        elif isinstance(reg, (int, float)):
            # small immediate should be b
            if self.b is None:
                self.b = reg
            elif self.b == reg:
                pass
            else:
                if isinstance(self.b, (int, float)):
                    raise AssembleError(f'Conflict small immediates {self.b} and small immediates {reg}')
                elif isinstance(self.b, Register):
                    raise AssembleError(f'Conflict register {self.b} and small immediates {reg}')
                elif isinstance(self.b, Signal):
                    raise AssembleError(f'Conflict signal {self.b} small immediates {reg}')
                else:
                    assert False, 'Bug: Lack of type exhaustivity'
        else:
            # already exist
            if self.a is not None:
                if self.a.name == reg.name:
                    return
            if self.b is not None and isinstance(self.b, Register):
                if self.b.name == reg.name:
                    return

            # not yet
            if reg.magic == 0:
                if self.a is None:
                    self.a = reg
                    return
                elif self.b is None:
                    self.b = reg
                    return
                else:
                    raise AssembleError('Too many register files read')
            elif reg.waddr < 6:
                return
            else:
                raise AssembleError(f'Invalid source register {reg}')

    def pack(self):
        raddr_a, raddr_b = 0, 0

        if isinstance(self.a, Register):
            raddr_a = self.a.waddr

        def pack_smimms_int(x):
            smimms_int = {}
            for i in range(16):
                smimms_int[i] = i
                smimms_int[i - 16] = i + 16
                smimms_int[float_to_int(2 ** (i - 8))] = i + 32
            return smimms_int[x]

        def pack_smimms_float(x):
            smimms_float = {}
            for i in range(16):
                # Denormal numbers
                smimms_float[int_to_float(i)] = i
                smimms_float[2 ** (i - 8)] = i + 32
            return smimms_float[x]

        if isinstance(self.b, int):
            raddr_b = pack_smimms_int(self.b)
        if isinstance(self.b, float):
            raddr_b = pack_smimms_float(self.b)
        if isinstance(self.b, Register):
            raddr_b = self.b.waddr
        if isinstance(self.b, Signal):
            raddr_b = pack_smimms_int(self.b.rot)

        return 0 \
            | (raddr_a << 6) \
            | (raddr_b << 0)


class ALUOp(object):

    OPERATIONS = {}
    MUX_A = {}
    MUX_B = {}

    def __init__(self, opr, dst=Instruction.REGISTERS['null'], src1=None, src2=None, cond=None, sig=None):

        assert opr in self.OPERATIONS

        self.name = opr
        self.op = self.OPERATIONS[opr]
        self.dst = dst
        self.src1 = src1
        self.src2 = src2
        self.cond = cond
        self.sigs = Signals()
        if sig is not None:
            self.sigs.add(sig)

        self.mux_a = None
        self.mux_b = None

        if self.name in self.MUX_A:
            self.mux_a = self.MUX_A[self.name]
        else:
            if self.src1 is None:
                raise AssembleError(f'"{self.name}" requires src1')
            if not isinstance(self.src1, (int, float, Register)):
                raise AssembleError(f'Invalid src1 object')

        if self.name in self.MUX_B:
            self.mux_b = self.MUX_B[self.name]
        else:
            if self.src2 is None:
                raise AssembleError(f'"{self.name}" requires src2')
            if not isinstance(self.src2, (int, float, Register)):
                raise AssembleError(f'Invalid src2 object')

    def pack(self, raddr):
        assert False, 'Bug: Not implemented'


class AddALUOp(ALUOp):

    def __init__(self, opr, *args, **kwargs):
        super(AddALUOp, self).__init__(opr, *args, **kwargs)

    def pack(self, raddr):

        op = self.op
        mux_a, mux_b = raddr.determine_mux((self.mux_a, self.mux_b), (self.src1, self.src2))

        if self.name in ['fadd', 'faddnf', 'fsub', 'fmax', 'fmin', 'fcmp']:

            op |= self.dst.pack_bits << 4

            a_unpack = self.src1.unpack_bits[0] if isinstance(self.src1, Register) else Register.INPUT_MODIFIER['none'][0]
            b_unpack = self.src2.unpack_bits[0] if isinstance(self.src2, Register) else Register.INPUT_MODIFIER['none'][0]

            ordering = a_unpack * 8 + mux_a > b_unpack * 8 + mux_b
            if (self.name in ['fmin', 'fadd'] and ordering) or (self.name in ['fmax', 'faddnf'] and not ordering):
                a_unpack, b_unpack = b_unpack, a_unpack
                mux_a, mux_b = mux_b, mux_a

            op |= a_unpack << 2
            op |= b_unpack << 0

        if self.name in ['fround', 'ftrunc', 'ffloor', 'fceil', 'fdx', 'fdy']:

            if self.src1.unpack_bits == Register.INPUT_MODIFIER['abs']:
                raise AssembleError('"abs" unpacking is not allowed here')

            mux_b |= self.dst.pack_bits
            a_unpack = self.src1.unpack_bits[0] if isinstance(self.src1, Register) else Register.INPUT_MODIFIER['none'][0]
            op = (op & ~(1 << 2)) | a_unpack << 2

        if self.name in ['ftoin', 'ftoiz', 'ftouz', 'ftoc']:

            if self.dst.pack_bits != Register.OUTPUT_MODIFIER['none']:
                raise AssembleError('packing is not allowed here')
            if self.src1.unpack_bits == Register.INPUT_MODIFIER['abs']:
                raise AssembleError('"abs" unpacking is not allowed here')

            a_unpack = self.src1.unpack_bits[0] if isinstance(self.src1, Register) else Register.INPUT_MODIFIER['none'][0]
            op &= ~0b1100
            op |= a_unpack << 2

        if self.name in ['vfpack']:

            a_unpack = self.src1.unpack_bits[0] if isinstance(self.src1, Register) else Register.INPUT_MODIFIER['none'][0]
            b_unpack = self.src2.unpack_bits[0] if isinstance(self.src2, Register) else Register.INPUT_MODIFIER['none'][0]

            op &= ~0b101
            op |= a_unpack << 2
            op |= b_unpack

        if self.name in ['vfmin', 'vfmax']:

            a_unpack = self.src1.unpack_bits[1] if isinstance(self.src1, Register) else Register.INPUT_MODIFIER['none'][1]
            op |= a_unpack

        return 0 \
            | (self.dst.magic << 44) \
            | (self.dst.waddr << 32) \
            | (op << 24) \
            | (mux_b << 15) \
            | (mux_a << 12)

    OPERATIONS = {
        # FADD is FADDNF depending on the order of the mux_a/mux_b.
        'fadd': 0,
        'faddnf': 0,
        'vfpack': 53,
        'add': 56,
        'sub': 60,
        'fsub': 64,
        'imin': 120,
        'imax': 121,
        'umin': 122,
        'umax': 123,
        'shl': 124,
        'shr': 125,
        'asr': 126,
        'ror': 127,
        # FMIN is FMAX depending on the order of the mux_a/mux_b.
        'fmin': 128,
        'fmax': 128,

        'vfmin': 176,

        'band': 181,
        'bor': 182,
        'bxor': 183,

        'bnot': 186,
        'neg': 186,
        'flapush': 186,
        'flbpush': 186,
        'flpop': 186,
        '_op_recip': 186,
        'setmsf': 186,
        'setrevf': 186,

        'nop': 187,
        'tidx': 187,
        'eidx': 187,
        'lr': 187,
        'vfla': 187,
        'vflna': 187,
        'vflb': 187,
        'vflnb': 187,
        'msf': 187,
        'revf': 187,
        'iid': 187,
        'sampid': 187,
        'barrierid': 187,
        'tmuwt': 187,
        'vpmwt': 187,

        '_op_rsqrt': 188,
        '_op_exp': 188,
        '_op_log': 188,
        '_op_sin': 188,
        '_op_rsqrt2': 188,

        'fcmp': 192,

        'vfmax': 240,

        'fround': 245,
        'ftoin': 245,
        'ftrunc': 245,
        'ftoiz': 245,
        'ffloor': 246,
        'ftouz': 246,
        'fceil': 246,
        'ftoc': 246,
        'fdx': 247,
        'fdy': 247,

        # The stvpms are distinguished by the waddr field.
        'stvpmv': 248,
        'stvpmd': 248,
        'stvpmp': 248,

        'itof': 252,
        'clz': 252,
        'utof': 252,
    }

    MUX_A = {
        'nop': 0,
        'tidx': 1,
        'eidx': 2,
        'lr': 3,
        'vfla': 4,
        'vflna': 5,
        'vflb': 6,
        'vflnb': 7,

        'msf': 0,
        'revf': 1,
        'iid': 2,
        'sampid': 3,
        'barrierid': 4,
        'tmuwt': 5,
        'vpmwt': 6,
    }

    MUX_B = {
        'bnot': 0,
        'neg': 1,
        'flapush': 2,
        'flbpush': 3,
        'flpop': 4,
        '_op_recip': 5,
        'setmsf': 6,
        'setrevf': 7,

        'nop': 0,
        'tidx': 0,
        'eidx': 0,
        'lr': 0,
        'vfla': 0,
        'vflna': 0,
        'vflb': 0,
        'vflnb': 0,

        'fround': 0,
        'ftoin': 3,
        'ftrunc': 4,
        'ftoiz': 7,
        'ffloor': 0,
        'ftouz': 3,
        'fceil': 4,
        'ftoc': 7,
        'fdx': 0,
        'fdy': 4,

        'msf': 2,
        'revf': 2,
        'iid': 2,
        'sampid': 2,
        'barrierid': 2,
        'tmuwt': 2,
        'vpmwt': 2,

        '_op_rsqrt': 3,
        '_op_exp': 4,
        '_op_log': 5,
        '_op_sin': 6,
        '_op_rsqrt2': 7,

        'itof': 0,
        'clz': 3,
        'utof': 4,
    }


class MulALUOp(ALUOp):

    def __init__(self, opr, *args, **kwargs):
        super(MulALUOp, self).__init__(opr, *args, **kwargs)

    def pack(self, raddr):

        op = self.op
        mux_a, mux_b = raddr.determine_mux((self.mux_a, self.mux_b), (self.src1, self.src2))

        if self.name in ['vfmul']:

            a_unpack = self.src1.unpack_bits[1] if isinstance(self.src1, Register) else Register.INPUT_MODIFIER['none'][1]
            op += a_unpack

        if self.name in ['fmul']:

            a_unpack = self.src1.unpack_bits[0] if isinstance(self.src1, Register) else Register.INPUT_MODIFIER['none'][0]
            b_unpack = self.src2.unpack_bits[0] if isinstance(self.src2, Register) else Register.INPUT_MODIFIER['none'][0]

            op += self.dst.pack_bits << 4
            op |= a_unpack << 2
            op |= b_unpack << 0

        if self.name in ['fmov']:

            a_unpack = self.src1.unpack_bits[0] if isinstance(self.src1, Register) else 0

            op |= (self.dst.pack_bits >> 1) & 1
            mux_b = ((self.dst.pack_bits & 1) << 2) | a_unpack

        return 0 \
            | (op << 58) \
            | (self.dst.magic << 45) \
            | (self.dst.waddr << 38) \
            | (mux_b << 21) \
            | (mux_a << 18)

    OPERATIONS = {
        'add': 1,
        'sub': 2,
        'umul24': 3,
        'vfmul': 4,
        'smul24': 9,
        'multop': 10,
        'fmov': 14,
        'nop': 15,
        'mov': 15,
        'fmul': 16,
    }

    MUX_A = {
        'nop': 0,
    }

    MUX_B = {
        'nop': 4,
        'mov': 7,
        'fmov': 0,
    }


class ALU(Instruction):

    def __init__(self, asm, opr, *args, **kwargs):
        super(ALU, self).__init__(asm)

        if opr in AddALUOp.OPERATIONS:
            self.add_op = AddALUOp(opr, *args, **kwargs)
            self.mul_op = None
        elif opr in MulALUOp.OPERATIONS:
            self.add_op = AddALUOp('nop')
            self.mul_op = MulALUOp(opr, *args, **kwargs)
        else:
            raise AssembleError(f'"{opr}" is unknown operation')
        self.repack()

    def dual_issue(self, opr, *args, **kwargs):
        if self.mul_op is not None:
            raise AssembleError('Conflict MulALU operation')
        self.mul_op = MulALUOp(opr, *args, **kwargs)
        self.repack()

    def __getattr__(self, name):
        if name in MulALUOp.OPERATIONS:
            return functools.partial(self.dual_issue, name)
        raise AssembleError(f'"{name}" is not MulALU operation')

    def repack(self):
        self.pack_result = None
        self.pack_result = self.pack()

    def pack(self):
        if self.pack_result is not None:
            return self.pack_result

        add_op = self.add_op
        mul_op = self.mul_op
        if mul_op is None:
            mul_op = MulALUOp('nop')

        raddr = ALURaddrs()
        raddr.add(add_op.src1)
        raddr.add(add_op.src2)
        raddr.add(mul_op.src1)
        raddr.add(mul_op.src2)

        sigs = Signals()
        sigs.add(add_op.sigs)
        sigs.add(mul_op.sigs)
        for sig in sigs:
            raddr.add(sig)

        if raddr.has_smimm() and not sigs.is_rotate():
            sigs.add(Instruction.SIGNALS['smimm'])

        cond = ALUConditions(add_op.cond, mul_op.cond)

        return 0 \
            | sigs.pack() \
            | cond.pack(sigs) \
            | add_op.pack(raddr) \
            | mul_op.pack(raddr) \
            | raddr.pack()


class Link(object):

    def __init__(self):
        pass


class Branch(Instruction):

    def __init__(self, asm, opr, src, *, cond, absolute=False, set_link=False):
        super(Branch, self).__init__(asm)

        self.cond_name = cond
        self.cond_br = {
            'always': 0,
            'a0': 2,
            'na0': 3,
            'alla': 4,
            'anyna': 5,
            'anya': 6,
            'allna': 7,
        }[self.cond_name]
        self.raddr_a = None
        self.addr_label = None
        self.addr = None
        self.set_link = set_link

        self.ub = 0
        self.bdu = 1

        if isinstance(src, Link):
            # Branch to link_reg
            self.bdi = 2
        elif isinstance(src, Register) and src.magic == 0:
            # Branch to reg
            self.bdi = 3
            self.raddr_a = src.waddr
        elif isinstance(src, Reference):
            # Branch to label
            self.bdi = 1
            self.addr_label = src
        elif isinstance(src, int):
            # Branch to imm
            self.bdi = 0 if absolute else 1
            self.addr = src
        else:
            raise AssembleError('Invalid src object')

    def unif_addr(self, src=None, absolute=False):
        self.ub = 1
        self.bdu = 1
        if src is None:
            self.bdu = 0 if absolute else 1
        elif isinstance(src, Register) and src.magic == 0:
            # Branch to reg
            self.bdu = 3
            if self.raddr_a is None or self.raddr_a == src.waddr:
                self.raddr_a = src.waddr
            else:
                raise AssembleError(f'Conflict registers')
        else:
            raise AssembleError('Invalid src object')

    def pack(self):

        if self.addr_label is not None:
            addr = int_to_uint((int(self.addr_label) - self.serial - 4) * 8)
        elif self.addr is not None:
            addr = self.addr
        else:
            addr = 0

        set_link = 1 if self.set_link else 0

        msfign = 0b00

        return 0 \
            | (0b10 << 56) \
            | (((addr & ((1 << 24) - 1)) >> 3) << 35) \
            | (self.cond_br << 32) \
            | ((addr >> 24) << 24) \
            | (set_link << 23) \
            | (msfign << 21) \
            | (self.bdu << 15) \
            | (self.ub << 14) \
            | (self.bdi << 12) \
            | ((self.raddr_a if self.raddr_a is not None else 0) << 6)


class Raw(Instruction):

    def __init__(self, asm, packed_code):
        super(Raw, self).__init__(asm)
        self.packed_code = packed_code

    def pack(self):
        return self.packed_code


class SFUIntegrator(Register):

    def __init__(self, asm, name):
        self.asm = asm
        self.reg_name = '_reg_'+name
        self.op_name = '_op_'+name
        reg = Instruction.REGISTERS[self.reg_name]
        super(SFUIntegrator, self).__init__(reg.name, reg.magic, reg.waddr)

    def __call__(self, dst, src, **kwargs):
        return ALU(self.asm, self.op_name, dst, src, **kwargs)


def _mov_a(asm, dst, src, **kwargs):
    return ALU(asm, 'bor', dst, src, src, **kwargs)


def _rotate_m(self, dst, src, rot, **kwargs):
    sigs = Signals()
    if 'sig' in kwargs:
        sigs.add(kwargs['sig'])
        del kwargs['sig']
    sigs.add(Instruction.SIGNALS['rot'](rot))
    if not isinstance(src, Register) or src.magic != 1:
        raise AssembleError('Invalid src object for rotate')
    return self.mov(dst, src, sig=sigs, **kwargs)


ALU.rotate = _rotate_m


def _quad_rotate_m(self, dst, src, rot, **kwargs):
    sigs = Signals()
    if 'sig' in kwargs:
        sigs.add(kwargs['sig'])
        del kwargs['sig']
    sigs.add(Instruction.SIGNALS['rot'](rot))
    if not isinstance(src, Register) or src.magic != 0:
        raise AssembleError('Invalid src object for quad_rotate')
    return self.mov(dst, src, sig=sigs, **kwargs)


ALU.quad_rotate = _quad_rotate_m


def _rotate_a(asm, *args, **kwargs):
    return ALU(asm, 'nop').rotate(*args, **kwargs)


def _quad_rotate_a(asm, *args, **kwargs):
    return ALU(asm, 'nop').quad_rotate(*args, **kwargs)


_alias_ops = {
    'mov': _mov_a,
    'rotate': _rotate_a,
    'quad_rotate': _quad_rotate_a,
}

_alias_regs = {
    'broadcast': Instruction.REGISTERS['r5rep'],
    'quad_broadcast': Instruction.REGISTERS['r5'],
}


class Loop(object):

    def __init__(self, asm, name):
        self.asm = asm
        self.name = name

    def b(self, *args, **kwargs):
        Branch(self.asm, 'b', Reference(self.asm, self.name), *args, **kwargs)


class LoopHelper(object):

    def __init__(self, asm):
        self.asm = asm

    def __enter__(self):
        name = self.asm._gen_unused_label('__generated_loop_label_{}')
        Label(self.asm).__getattr__(name)
        return Loop(self.asm, name)

    def __exit__(self, ex_type, ex_value, trace):
        pass


def qpu(func):

    @functools.wraps(func)
    def decorator(asm, *args, **kwargs):
        g = func.__globals__
        g_orig = g.copy()
        g['L'] = Label(asm)
        g['R'] = Reference(asm)
        g['loop'] = LoopHelper(asm)
        g['b'] = functools.partial(Branch, asm, 'b')
        g['link'] = Link()
        g['raw'] = functools.partial(Raw, asm)
        g['namespace'] = functools.partial(LabelNameSpace, asm)
        for mul_op in MulALUOp.OPERATIONS.keys():
            g[mul_op] = functools.partial(ALU, asm, mul_op)
        for add_op in AddALUOp.OPERATIONS.keys():
            if not add_op.startswith('_op_'):
                g[add_op] = functools.partial(ALU, asm, add_op)
        for waddr, reg in Instruction.REGISTERS.items():
            if waddr.startswith('_reg_'):
                g[waddr[5:]] = SFUIntegrator(asm, waddr[5:])
            else:
                g[waddr] = reg
        g['rf'] = [Instruction.REGISTERS[f'rf{i}'] for i in range(64)]
        for alias_name, alias_op in _alias_ops.items():
            g[alias_name] = functools.partial(alias_op, asm)
        for alias_name, alias_reg in _alias_regs.items():
            g[alias_name] = alias_reg
        for name, sig in Instruction.SIGNALS.items():
            if name != 'smimm':  # smimm signal is automatically derived
                g[name] = sig
        func(asm, *args, **kwargs)
        g.clear()
        for key, value in g_orig.items():
            g[key] = value

    return decorator
