#!/usr/bin/env python3

import struct
import functools
import numpy as np
from videocore6 import float_to_int, int_to_float, int_to_uint


class AssembleError(Exception):
    pass


class Assembly(list):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # name : index
        self.labels = {}

    def finalize(self):
        for idx, insn in enumerate(self):
            if insn.insn_type == 'branch':
                if insn.addr_label is not None:
                    idx_addr = self.labels[insn.addr_label]
                    insn.addr = int_to_uint((idx_addr - idx - 4) * 8)
            insn.finalized = True


class Label(object):

    def __init__(self, asm):
        self.asm = asm

    def __getattr__(self, name):
        if name in self.asm.labels:
            raise AssembleError(f'Label is duplicated: {name}')
        self.asm.labels[name] = len(self.asm)


class Reference(str):

    def __getattr__(self, name):
        return name


class Instruction(object):

    # name : (magic, waddr)
    waddrs = {
            'null' : (1, 6),
            'tlb' : (1, 7),
            'tlbu' : (1, 8),
            'tmu' : (1, 9),
            'tmul' : (1, 10),
            'tmud' : (1, 11),
            'tmua' : (1, 12),
            'tmuau' : (1, 13),
            'vpm' : (1, 14),
            'vpmu' : (1, 15),
            'sync' : (1, 16),
            'syncu' : (1, 17),
            'syncb' : (1, 18),
            'recip' : (1, 19),
            'rsqrt' : (1, 20),
            'exp' : (1, 21),
            'log' : (1, 22),
            'sin' : (1, 23),
            'rsqrt2' : (1, 24),
            'tmuc' : (1, 32),
            'tmus' : (1, 33),
            'tmut' : (1, 34),
            'tmur' : (1, 35),
            'tmui' : (1, 36),
            'tmub' : (1, 37),
            'tmudref' : (1, 38),
            'tmuoff' : (1, 39),
            'tmuscm' : (1, 40),
            'tmusf' : (1, 41),
            'tmuslod' : (1, 42),
            'tmuhs' : (1, 43),
            'tmuhscm' : (1, 44),
            'tmuhsf' : (1, 45),
            'tmuhslod' : (1, 46),
            'r5rep' : (1, 55),
    }
    for i in range(6):
        waddrs[f'r{i}'] = (1, i)
    for i in range(64):
        waddrs[f'rf{i}'] = (0, i)

    add_ops = {
            'add' : 56,
            'sub' : 60,
            'imin' : 120,
            'imax' : 121,
            'umin' : 122,
            'umax' : 123,
            'shl' : 124,
            'shr' : 125,
            'asr' : 126,
            'ror' : 127,
            'band' : 181,
            'bor' : 182,
            'bxor' : 183,

            'bnot' : 186,
            'neg' : 186,
            'flapush' : 186,
            'flbpush' : 186,
            'flpop' : 186,
            'op_recip' : 186,
            'setmsf' : 186,
            'setrevf' : 186,

            'nop' : 187,
            'tidx' : 187,
            'eidx' : 187,
            'lr' : 187,
            'vfla' : 187,
            'vflna' : 187,
            'vflb' : 187,
            'vflnb' : 187,
            'msf' : 187,
            'revf' : 187,
            'iid' : 187,
            'sampid' : 187,
            'barrierid' : 187,
            'tmuwt' : 187,
            'vpmwt' : 187,

            'op_rsqrt' : 188,
            'op_exp' : 188,
            'op_log' : 188,
            'op_sin' : 188,
            'op_rsqrt2' : 188,

            # The stvpms are distinguished by the waddr field.
            'stvpmv' : 248,
            'stvpmd' : 248,
            'stvpmp' : 248,

            'clz' : 252,
    }
    for i in range(0, 48):
        # FADD is FADDNF depending on the order of the mux_a/mux_b.
        add_ops[f'fadd{i}'] = i
        add_ops[f'faddnf{i}'] = i
    for i in range(64, 112):
        add_ops[f'fsub{i}'] = i
    for i in range(128, 176):
        add_ops[f'fmin{i}'] = i
        add_ops[f'fmax{i}'] = i

    add_op_mux_a = {
            'nop' : 0,
            'tidx' : 1,
            'eidx' : 2,
            'lr' : 3,
            'vfla' : 4,
            'vflna' : 5,
            'vflb' : 6,
            'vflnb' : 7,

            'msf' : 0,
            'revf' : 1,
            'iid' : 2,
            'sampid' : 3,
            'barrierid' : 4,
            'tmuwt' : 5,
            'vpmwt' : 6,
    }

    add_op_mux_b = {
            'bnot' : 0,
            'neg' : 1,
            'flapush' : 2,
            'flbpush' : 3,
            'flpop' : 4,
            'op_recip' : 5,
            'setmsf' : 6,
            'setrevf' : 7,

            'nop' : 0,
            'tidx' : 0,
            'eidx' : 0,
            'lr' : 0,
            'vfla' : 0,
            'vflna' : 0,
            'vflb' : 0,
            'vflnb' : 0,

            'msf' : 2,
            'revf' : 2,
            'iid' : 2,
            'sampid' : 2,
            'barrierid' : 2,
            'tmuwt' : 2,
            'vpmwt' : 2,

            'op_rsqrt' : 3,
            'op_exp' : 4,
            'op_log' : 5,
            'op_sin' : 6,
            'op_rsqrt2' : 7,

            'clz' : 3,
    }

    mul_ops = {
            'add' : 1,
            'sub' : 2,
            'umul24' : 3,
            'smul24' : 9,
            'multop' : 10,
            'fmov' : 14,

            'fmov_0' : 15,
            'fmov_1' : 15,
            'fmov_2' : 15,
            'fmov_3' : 15,
            'nop' : 15,
            'mov' : 15,
    }
    for i in range(16, 64):
        mul_ops[f'fmul{i}'] = i

    mul_op_mux_a = {
            'nop' : 0,
    }

    mul_op_mux_b = {
            'fmov_0' : 0,
            'fmov_1' : 1,
            'fmov_2' : 2,
            'fmov_3' : 3,
            'nop' : 4,
            'mov' : 7,
    }

    # Don't ask me why...
    def sig_to_num(self):
        sigs = {
                frozenset() : 0,
                frozenset(['thrsw']) : 1,
                frozenset(['ldunif']) : 2,
                frozenset(['thrsw', 'ldunif']) : 3,
                frozenset(['ldtmu']) : 4,
                frozenset(['thrsw', 'ldtmu']) : 5,
                frozenset(['ldtmu', 'ldunif']) : 6,
                frozenset(['thrsw', 'ldtmu', 'ldunif']) : 7,
                frozenset(['ldvary']) : 8,
                frozenset(['thrsw', 'ldvary']) : 9,
                frozenset(['ldvary', 'ldunif']) : 10,
                frozenset(['thrsw', 'ldvary', 'ldunif']) : 11,
                frozenset(['ldunifrf']) : 12,
                frozenset(['thrsw', 'ldunifrf']) : 13,
                frozenset(['smimm', 'ldvary']) : 14,
                frozenset(['smimm']) : 15,
                frozenset(['ldtlb']) : 16,
                frozenset(['ldtlbu']) : 17,
                frozenset(['wrtmuc']) : 18,
                frozenset(['thrsw', 'wrtmuc']) : 19,
                frozenset(['ldvary', 'wrtmuc']) : 20,
                frozenset(['thrsw', 'ldvary', 'wrtmuc']) : 21,
                frozenset(['ucb']) : 22,
                frozenset(['rot']) : 23,
                frozenset(['ldunifa']) : 24,
                frozenset(['ldunifarf']) : 25,
                frozenset(['smimm', 'ldtmu']) : 31,
        }
        return sigs[frozenset(self.sig)]

    def sig_writes(self):
        # XXX: What if multiple loads are issued?
        return len(self.sig.intersection([
                'ldunifrf', 'ldunifarf', 'ldvary', 'ldtmu', 'ldtlb', 'ldtlbu'
        ])) > 0

    def cond_to_num(self):
        if self.sig_writes():
            return (self.sig_magic << 6) | self.sig_waddr

        conds_push = {
                'pushz' : 1,
                'pushn' : 2,
                'pushc' : 3,
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
                'ifa' : 0,
                'ifb' : 1,
                'ifna' : 2,
                'ifnb' : 3,
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
            [         0, 0b0100000,         0,         0 ], # none
            [ 0b0110000, 0b1000000, 0b0110000, 0b1000000 ], # mul_insn
            [ 0b0010000, 0b0100000,      None,      None ], # mul_push
            [ 0b0010000,      None,      None,      None ], # mul_update
        ][mul_cond][add_cond]

        assert(result is not None)

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

        return result


    def cond_br_to_num(self):
        return {
                'always' : 0,
                'a0' : 2,
                'na0' : 3,
                'alla' : 4,
                'anyna' : 5,
                'anya' : 6,
                'allna' : 7,
        }[self.cond_br]

    def __init__(self, asm, opr, *args, **kwargs):
        asm.append(self)

        self.insn_type = 'alu'
        self.finalized = False

        self.sig = set()
        self.sig_magic, self.sig_waddr = self.waddrs['null']

        self.cond_add = None
        self.cond_mul = None

        self.op_add = self.add_ops['nop']
        self.ma, self.waddr_a = self.waddrs['null']
        self.add_a = self.add_op_mux_a['nop']
        self.add_b = self.add_op_mux_b['nop']

        self.op_mul = self.mul_ops['nop']
        self.mm, self.waddr_m = self.waddrs['null']
        self.mul_a = self.mul_op_mux_a['nop']
        self.mul_b = self.mul_op_mux_b['nop']

        self.raddr_a = None
        self.raddr_b = None

        self.addr = 0
        self.addr_label = None

        if opr in self.add_ops:
            self.AddALU(self, opr, *args, **kwargs)
        elif opr == 'b':
            self.Branch(self, opr, *args, **kwargs)

    def __getattr__(self, name):
        if self.insn_type == 'alu' and name in Instruction.mul_ops.keys():
            return functools.partial(self.MulALU, self, name)
        else:
            raise AttributeError(name)

    def __int__(self):
        if not self.finalized:
            raise ValueError('Not yet finalized')
        if self.insn_type == 'alu':
            return (self.op_mul << 58) \
                    | (self.sig_to_num() << 53) \
                    | (self.cond_to_num() << 46) \
                    | (self.mm << 45) \
                    | (self.ma << 44) \
                    | (self.waddr_m << 38) \
                    | (self.waddr_a << 32) \
                    | (self.op_add << 24) \
                    | (self.mul_b << 21) \
                    | (self.mul_a << 18) \
                    | (self.add_b << 15) \
                    | (self.add_a << 12) \
                    | ((self.raddr_a if self.raddr_a is not None else 0) << 6) \
                    | ((self.raddr_b if self.raddr_b is not None else 0) << 0)
        elif self.insn_type == 'branch':
            return (0b10 << 56) \
                    | (((self.addr & ((1 << 24) - 1)) >> 3) << 35) \
                    | (self.cond_br_to_num() << 32) \
                    | ((self.addr >> 24) << 24) \
                    | (self.bdi << 12) \
                    | ((self.raddr_a if self.raddr_a is not None else 0) << 6)

    class ALU(object):

        # XXX: Type-strict dictionary

        smimms_int = {}
        for i in range(16):
            smimms_int[i] = i
            smimms_int[i - 16] = i + 16
            smimms_int[float_to_int(2 ** (i - 8))] = i + 32

        smimms_float = {}
        for i in range(16):
            # Denormal numbers
            smimms_float[int_to_float(i)] = i
            smimms_float[2 ** (i - 8)] = i + 32

        def manage_src(self, insn, src):

            is_smimm_int = isinstance(src, int)
            is_smimm_float = isinstance(src, float)

            if is_smimm_int or is_smimm_float:
                if is_smimm_int:
                    rb = self.smimms_int[src]
                elif is_smimm_float:
                    rb = self.smimms_float[src]
                if insn.raddr_b is None:
                    insn.raddr_b = rb
                    insn.sig.add('smimm')
                else:
                    if 'smimm' not in insn.sig:
                        raise AssembleError('Too many requests for raddr_b')
                    elif insn.raddr_b != rb:
                        raise AssembleError('Small immediates conflict')
                return 7

            if src.startswith('rf'):
                idx = int(src[2:])
                assert 0 <= idx <= 63
                if insn.raddr_a in [None, idx]:
                    insn.raddr_a = idx
                    return 6
                elif 'smimm' not in insn.sig and insn.raddr_b in [None, idx]:
                    insn.raddr_b = idx
                    return 7
                else:
                    raise AssembleError('Too many register files read')
            elif src.startswith('r'):
                idx = int(src[1:])
                assert 0 <= idx <= 5
                return idx
            else:
                raise AssembleError(f'Unknown source register {src}')

        def __init__(self, insn, opr, dst, src1 = None, src2 = None,
                cond = None, sig = None):
            # XXX: With Python >= 3.8 we can use positional-only params.
            if src1 is None and src2 is not None:
                raise AssembleError('src2 is specified while src1 is not')

            insn.insn_type = 'alu'

            if sig is not None:
                if isinstance(sig, str):
                    insn.sig = insn.sig.union(sig.split())
                elif isinstance(sig, (list, tuple)):
                    insn.sig = insn.sig.union(sig)
                else:
                    raise AssembleError('Unknown object specified for sig')

            self.cond = cond

            self.op = self.ops[opr]

            if insn.sig_writes():
                if (self.cond is not None and self.cond != '') or \
                        (insn.cond_add is not None and insn.cond_add != ''):
                    raise AssembleError('cond must be none with sig write')
                # XXX: Don't specify dest regs to both add and mul for now!
                insn.sig_magic, insn.sig_waddr = Instruction.waddrs[dst]
                return

            self.magic, self.waddr = Instruction.waddrs[dst]

            if src1 is None:
                self.mux_a = self.op_mux_a[opr]
            else:
                self.mux_a = self.manage_src(insn, src1)

            if src2 is None:
                self.mux_b = self.op_mux_b[opr]
            else:
                self.mux_b = self.manage_src(insn, src2)


    class AddALU(ALU):

        def __init__(self, insn, *args, **kwargs):
            self.ops = insn.add_ops
            self.op_mux_a = insn.add_op_mux_a
            self.op_mux_b = insn.add_op_mux_b
            super().__init__(insn, *args, **kwargs)
            insn.cond_add = self.cond
            insn.op_add = self.op
            if not insn.sig_writes():
                insn.ma = self.magic
                insn.waddr_a = self.waddr
                insn.add_a = self.mux_a
                insn.add_b = self.mux_b


    class MulALU(ALU):

        def __init__(self, insn, *args, **kwargs):
            self.ops = insn.mul_ops
            self.op_mux_a = insn.mul_op_mux_a
            self.op_mux_b = insn.mul_op_mux_b
            super().__init__(insn, *args, **kwargs)
            insn.cond_mul = self.cond
            insn.op_mul = self.op
            if not insn.sig_writes():
                insn.mm = self.magic
                insn.waddr_m = self.waddr
                insn.mul_a = self.mux_a
                insn.mul_b = self.mux_b

    class Branch(object):

        def __init__(self, insn, opr, src, *, cond):

            insn.insn_type = 'branch'
            insn.cond_br = cond

            if src.startswith('rf'):
                insn.bdi = 3
                insn.raddr_a = int(src[2:])
            else:
                # Branch to label
                insn.bdi = 1
                insn.addr_label = src

def mov(asm, dst, src, **kwargs):
    return Instruction(asm, 'bor', dst, src, src, **kwargs)

_alias_ops = [
    mov,
    # TODO: impl these aliases for happy programming
    # fadd,
    # faddnf,
    # fsub,
    # fmin,
    # fmax,
    # fmul,
]

def qpu(func):

    @functools.wraps(func)
    def decorator(asm, *args, **kwargs):
        g = func.__globals__
        g_orig = g.copy()
        g['L'] = Label(asm)
        g['R'] = Reference()
        g['b'] = functools.partial(Instruction, asm, 'b')
        for add_op in Instruction.add_ops.keys():
            g[add_op] = functools.partial(Instruction, asm, add_op)
        for alias_op in _alias_ops:
            g[alias_op.__name__] = functools.partial(alias_op, asm)
        for waddr in Instruction.waddrs.keys():
            g[waddr] = waddr
        func(asm, *args, **kwargs)
        g.clear()
        for key, value in g_orig.items():
            g[key] = value

    return decorator
